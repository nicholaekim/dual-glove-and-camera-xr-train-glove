"""Live LeapC tracking behind the same `drain()` API as the glove receiver.

`LeapStream` owns a `leap.Connection` and a `leap.Listener`. The bindings run
their own polling thread, so tracking callbacks arrive off-thread; each hand
is converted to a `LeapHand` there and pushed onto a bounded queue, and the
main thread (viewer, recorder, whatever) drains it on its own schedule. That
is exactly how `xr_hand.receiver.OSCHandReceiver` works, down to the item
shape:

    glove:  ("right", [187 raw floats])
    leap:   ("right", LeapHand)

so a script can hold either object in the same variable and loop over
`source.drain(64)` without caring which sensor is behind it.

Nothing imports `leap` until `start()` runs. When the bindings or the device
are missing, `start()` raises `LeapUnavailable` with the next command to
type — never a traceback out of the SDK, because on this machine (no
Ultraleap software, no camera) that is the normal case, and `--mock` is the
answer.
"""
import logging
import time
from queue import Empty, Full, Queue
from typing import List, Optional, Tuple

from .to_openxr import leap_hand_from_api
from .types import LeapHand

log = logging.getLogger(__name__)

# Same shape as xr_hand.receiver.QueueItem: (hand_side, payload).
QueueItem = Tuple[str, LeapHand]

SETUP_SCRIPT = r"powershell -ExecutionPolicy Bypass -File scripts\leap\setup_bindings.ps1"
CHECK_SCRIPT = r"python scripts\leap\check_setup.py"

TRACKING_MODES = ("desktop", "hmd", "screentop")


class LeapUnavailable(RuntimeError):
    """No usable Ultraleap tracking: bindings missing, or no device.

    The message always ends with the exact next step, because this is the
    error every new machine hits first.
    """


class LeapStream:
    """Live Ultraleap tracking as `(hand_side, LeapHand)` items.

    Args:
        mode: tracking mode — 'desktop' (module flat on the table, lenses up,
            which is the protocol in docs/ultraleap_ir170_plan.md), 'hmd' or
            'screentop'.
        device_timeout: seconds `start()` waits for a device to appear before
            giving up with `LeapUnavailable`.
        queue_maxsize: hands buffered before the oldest are dropped. At 90 Hz
            with two hands, 480 is about 2.7 s of slack.
    """

    def __init__(
        self,
        mode: str = "desktop",
        device_timeout: float = 5.0,
        queue_maxsize: int = 480,
    ):
        if mode not in TRACKING_MODES:
            raise ValueError(f"mode must be one of {TRACKING_MODES}, got {mode!r}")
        self.mode = mode
        self.device_timeout = float(device_timeout)
        self.queue: "Queue[QueueItem]" = Queue(maxsize=queue_maxsize)

        self._leap = None
        self._connection = None
        self._listener = None

        # observable state, for HUDs and stats
        self.connected = False
        self.device_serial: Optional[str] = None
        self.frames = 0
        self.hands_seen = 0
        self.dropped = 0
        self.framerate = 0.0
        self.last_event_time: Optional[float] = None

    # --- SDK loading ----------------------------------------------------
    @staticmethod
    def import_leap():
        """Import the `leap` bindings or raise `LeapUnavailable` with the fix."""
        try:
            import leap  # noqa: PLC0415 — optional dependency, imported lazily
        except ImportError as e:
            raise LeapUnavailable(
                "the Ultraleap Python bindings are not installed in this "
                f"interpreter (import leap failed: {e}).\n"
                f"  Next step: {SETUP_SCRIPT}\n"
                "  (it builds leapc_cffi against the LeapSDK and installs "
                "leapc-python-api), then verify with:\n"
                f"  {CHECK_SCRIPT}\n"
                "  No camera to hand? Every script here takes --mock."
            ) from e
        return leap

    def _tracking_mode(self):
        leap = self._leap
        return {
            "desktop": leap.TrackingMode.Desktop,
            "hmd": leap.TrackingMode.HMD,
            "screentop": leap.TrackingMode.ScreenTop,
        }[self.mode]

    # --- listener -------------------------------------------------------
    def _build_listener(self):
        """Subclass `leap.Listener` now that the module is actually imported."""
        leap = self._leap
        stream = self
        # LeapCannotOpenDeviceError lives in leap.exceptions, not on the
        # package root; fall back to the base LeapError if that ever moves.
        cannot_open = getattr(leap.exceptions, "LeapCannotOpenDeviceError",
                              leap.LeapError)

        class _StreamListener(leap.Listener):
            def on_connection_event(self, event):
                stream.connected = True
                log.info("connected to the Ultraleap tracking service")

            def on_connection_lost_event(self, event):
                stream.connected = False
                log.warning("lost the Ultraleap tracking service connection")

            def on_device_event(self, event):
                # The bindings' own example: open the device for its info, and
                # fall back to reading it unopened when the device is already
                # held (LeapCannotOpenDeviceError).
                try:
                    with event.device.open():
                        info = event.device.get_info()
                except cannot_open:
                    try:
                        info = event.device.get_info()
                    except Exception as e:      # pragma: no cover - hardware path
                        log.warning("device found but its info is unreadable "
                                    "(is another app using the camera?): %s", e)
                        stream.device_serial = "unknown"
                        return
                except Exception as e:          # pragma: no cover - hardware path
                    log.warning("device found but could not be opened: %s", e)
                    stream.device_serial = "unknown"
                    return
                stream.device_serial = getattr(info, "serial", "unknown")
                log.info("device: %s", stream.device_serial)

            def on_tracking_event(self, event):
                stream._on_tracking(event)

        return _StreamListener()

    def _on_tracking(self, event) -> None:
        # Frame age at receipt (not end-to-end latency): how old the tracking
        # data already was when this callback ran, on the LeapC clock.
        age_us = None
        try:
            age_us = float(self._leap.get_now() - event.timestamp)
        except Exception:                        # pragma: no cover - hardware path
            pass

        self.frames += 1
        self.last_event_time = time.time()
        rate = getattr(event, "framerate", None)
        if rate:
            self.framerate = float(rate)

        for hand in event.hands:
            try:
                lh = leap_hand_from_api(hand, event, frame_age_us=age_us)
            except Exception as e:               # pragma: no cover - hardware path
                log.warning("could not convert a hand from frame %s: %s",
                            getattr(event, "tracking_frame_id", "?"), e)
                continue
            self.hands_seen += 1
            self._enqueue((lh.hand_side, lh))

    def _enqueue(self, item: QueueItem) -> None:
        try:
            self.queue.put_nowait(item)
        except Full:
            # Same policy as the glove receiver: drop the oldest hand rather
            # than block the SDK's polling thread.
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(item)
            except (Empty, Full):
                pass
            self.dropped += 1

    # --- lifecycle ------------------------------------------------------
    def start(self) -> None:
        """Connect, set the tracking mode, and wait for a device.

        Raises:
            LeapUnavailable: bindings missing, service not running, or no
                device within `device_timeout` seconds.
        """
        self._leap = self.import_leap()
        self._connection = self._leap.Connection()
        self._listener = self._build_listener()
        self._connection.add_listener(self._listener)

        try:
            # connect() starts the bindings' polling thread; callbacks run on
            # it. The tracking mode can only be set once the connection is up.
            self._connection.connect()
        except Exception as e:
            raise LeapUnavailable(
                f"could not connect to the Ultraleap tracking service ({e}).\n"
                "  Next step: open the Ultraleap Control Panel and confirm the "
                "service is running, then:\n"
                f"  {CHECK_SCRIPT}"
            ) from e

        deadline = time.time() + self.device_timeout
        while time.time() < deadline:
            if self.device_serial is not None or self.frames:
                break
            time.sleep(0.05)
        else:
            self.stop()
            raise LeapUnavailable(
                f"no Ultraleap device appeared within {self.device_timeout:g} s.\n"
                "  Next step: plug the Stereo IR 170 into a direct USB port "
                "(not a hub), open the Ultraleap Control Panel and confirm the "
                "device is listed and the tracking service is running, then:\n"
                f"  {CHECK_SCRIPT}\n"
                "  No camera to hand? Every script here takes --mock."
            )

        try:
            self._connection.set_tracking_mode(self._tracking_mode())
        except Exception as e:                   # pragma: no cover - hardware path
            log.warning("could not set tracking mode %s: %s", self.mode, e)

    def stop(self) -> None:
        """Disconnect. Safe to call twice, and safe if `start()` failed."""
        if self._connection is not None:
            try:
                self._connection.disconnect()
            except Exception as e:               # pragma: no cover - hardware path
                log.debug("disconnect complained (ignored): %s", e)
            self._connection = None
        self._listener = None
        self.connected = False

    @property
    def connection(self):
        """The live `leap.Connection`, or None before `start()` / after `stop()`.

        Only `replay.RawRecording` needs this, to attach LeapC's own `.lmt`
        recorder alongside our JSONL.
        """
        return self._connection

    # --- consumer side --------------------------------------------------
    def drain(self, max_items: int = 16) -> List[QueueItem]:
        """Up to `max_items` pending (hand_side, LeapHand), oldest first."""
        items: List[QueueItem] = []
        for _ in range(max_items):
            try:
                items.append(self.queue.get_nowait())
            except Empty:
                break
        return items


def open_stream(
    mock: bool = False,
    mode: str = "desktop",
    device_timeout: float = 5.0,
    pose: Optional[str] = None,
):
    """The one line every script uses: a real stream, or the mock.

    Returns an already-started source with `drain()`, `stop()` and the stats
    attributes the HUDs read. `pose` is honoured by the mock only.
    """
    if mock:
        from .mock import MockLeapStream
        source = MockLeapStream(pose=pose)
    else:
        source = LeapStream(mode=mode, device_timeout=device_timeout)
    source.start()
    return source
