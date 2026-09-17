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
type — never a traceback out of the SDK, because on a machine without the
camera that is the normal case, and `--mock` is the answer.

Adding policy flags later (images, for the IR evidence the Phase 2 gate
wants): `PolicyFlag` is **not** exported from the package root. It is
`leap.enums.PolicyFlag`, and it goes through
`connection.set_policy_flags(flags_to_set=[leap.enums.PolicyFlag.Images])`,
which waits for a Policy event and returns the flags now in force.
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
        mode_timeout: seconds allowed for the tracking-mode round trip.
        queue_maxsize: hands buffered before the oldest are dropped. At 90 Hz
            with two hands, 480 is about 2.7 s of slack.
    """

    def __init__(
        self,
        mode: str = "desktop",
        device_timeout: float = 5.0,
        mode_timeout: float = 5.0,
        queue_maxsize: int = 480,
    ):
        if mode not in TRACKING_MODES:
            raise ValueError(f"mode must be one of {TRACKING_MODES}, got {mode!r}")
        self.mode = mode
        self.device_timeout = float(device_timeout)
        self.mode_timeout = float(mode_timeout)
        self.queue: "Queue[QueueItem]" = Queue(maxsize=queue_maxsize)

        self._leap = None
        self._connection = None
        self._listener = None
        self._warned_device = False

        # observable state, for HUDs and stats
        self.connected = False
        self.tracking_mode_confirmed = False
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
                # The serial can only be read from an OPEN device: get_info()
                # on a closed one raises DeviceNotOpenException, so there is no
                # fallback to reading it unopened. A device we cannot open is
                # still a device the service is tracking with, so record the
                # serial as unavailable, say so once, and carry on.
                try:
                    with event.device.open():
                        info = event.device.get_info()
                except cannot_open as e:
                    stream._device_unavailable(
                        "could not open the device to read its serial "
                        f"(is another app using the camera?): {e}")
                    return
                except Exception as e:          # pragma: no cover - hardware path
                    stream._device_unavailable(f"device info unreadable: {e}")
                    return
                stream.device_serial = getattr(info, "serial", "unavailable")
                log.info("device: %s", stream.device_serial)

            def on_tracking_event(self, event):
                stream._on_tracking(event)

        return _StreamListener()

    def _device_unavailable(self, why: str) -> None:
        """A device is there but its serial is not readable. Warn once."""
        self.device_serial = "unavailable"
        if not self._warned_device:
            self._warned_device = True
            log.warning("%s - tracking continues; the serial will read "
                        "'unavailable' in recordings", why)

    def _on_tracking(self, event) -> None:
        # Two clocks, read in one place because they have to agree.
        #
        #   age_us        frame age at receipt (not end-to-end latency): how
        #                 old the tracking data already was when this callback
        #                 ran, on the LeapC clock, whose epoch is arbitrary.
        #   capture_time  the same instant on the WALL clock, by subtracting
        #                 that age from time.time() here.
        #
        # This callback is the only place the two clocks can be related,
        # because it is the only moment we hold both. Doing it here is what
        # lets every consumer downstream have a camera timestamp comparable
        # with the glove's, rather than the time we happened to drain the
        # queue — hands arrive from the polling thread in bursts, so drain
        # time says nothing about when a hand was in a pose.
        now = time.time()
        age_us = capture_time = None
        try:
            age_us = float(self._leap.get_now() - event.timestamp)
            capture_time = now - age_us / 1e6
        except Exception:                        # pragma: no cover - hardware path
            pass

        self.frames += 1
        self.last_event_time = now
        rate = getattr(event, "framerate", None)
        if rate:
            self.framerate = float(rate)

        for hand in event.hands:
            try:
                lh = leap_hand_from_api(hand, event, frame_age_us=age_us,
                                        capture_time=capture_time)
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
        """Connect, set the tracking mode, wait for a device, verify the mode.

        The order matters. The tracking mode is set the moment the connection
        is up and the queue is emptied straight after, so not one frame
        captured in some previous session's mode can reach a recording.

        Raises:
            LeapUnavailable: bindings missing, service not running, no device
                within `device_timeout` seconds, or the tracking mode could
                not be set and confirmed. Whatever the reason, the LeapC
                connection is closed before the error leaves this method.
        """
        self._leap = self.import_leap()
        self._warned_device = False
        # response_timeout bounds the mode round trip below; the default of
        # 10 s is a long time to sit in front of a camera wondering.
        self._connection = self._leap.Connection(response_timeout=self.mode_timeout)
        self._listener = self._build_listener()
        self._connection.add_listener(self._listener)

        try:
            # connect() starts the bindings' polling thread; callbacks run on
            # it. The tracking mode can only be set once the connection is up.
            self._connection.connect()
        except Exception as e:
            self._safe_disconnect()
            raise LeapUnavailable(
                f"could not connect to the Ultraleap tracking service ({e}).\n"
                "  Next step: open the Ultraleap Control Panel and confirm the "
                "service is running, then:\n"
                f"  {CHECK_SCRIPT}"
            ) from e

        # Everything from here can fail, and none of it may leave a connection
        # open behind it.
        try:
            self._set_tracking_mode()
            self._discard_pending()
            self._wait_for_device()
            self._verify_tracking_mode()
            # Frames that arrived between the set and its confirmation were
            # queued in an unconfirmed mode; only what follows counts.
            self._discard_pending()
        except BaseException:
            self._safe_disconnect()
            raise

    def _set_tracking_mode(self) -> None:
        try:
            self._connection.set_tracking_mode(self._tracking_mode())
        except Exception as e:
            raise LeapUnavailable(
                f"could not set the {self.mode} tracking mode ({e}).\n"
                "  Recording in the wrong mode would put every hand in the "
                "wrong frame, so this is fatal rather than a warning.\n"
                "  Next step: open the Ultraleap Control Panel, set the "
                "tracking mode there, then:\n"
                f"  {CHECK_SCRIPT}"
            ) from e

    def _discard_pending(self) -> None:
        """Drop anything captured before the mode was set, and start counting.

        `device_serial` survives: the device event fires once, on connect, and
        the device's identity does not depend on the tracking mode.
        """
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break
        self.frames = 0
        self.hands_seen = 0
        self.dropped = 0
        self.framerate = 0.0
        self.last_event_time = None

    def _wait_for_device(self) -> None:
        deadline = time.time() + self.device_timeout
        while time.time() < deadline:
            if self.device_serial is not None or self.frames:
                return
            time.sleep(0.05)
        raise LeapUnavailable(
            f"no Ultraleap device appeared within {self.device_timeout:g} s.\n"
            "  Next step: plug the Stereo IR 170 into a direct USB port "
            "(not a hub), open the Ultraleap Control Panel and confirm the "
            "device is listed and the tracking service is running, then:\n"
            f"  {CHECK_SCRIPT}\n"
            "  No camera to hand? Every script here takes --mock."
        )

    def _verify_tracking_mode(self) -> None:
        """Confirm the server really is in the mode we asked for.

        This runs after the device wait, not straight after the set, because
        `get_tracking_mode()` waits for a TrackingMode event and the service
        does not emit one while no device is attached (measured: it times out
        after the full response timeout). Verifying earlier would turn every
        "no camera plugged in" into a slow, misleading mode error. The set
        itself already happened before any frame was counted, which is what
        keeps the data honest; this is the confirmation.
        """
        want = self._tracking_mode()
        try:
            got = self._connection.get_tracking_mode()
        except Exception as e:
            raise LeapUnavailable(
                f"a device is tracking but the service never confirmed the "
                f"{self.mode} tracking mode ({type(e).__name__}: {e}).\n"
                "  Next step: open the Ultraleap Control Panel, check the "
                "tracking mode and that the device is healthy, then:\n"
                f"  {CHECK_SCRIPT}"
            ) from e
        if got != want:
            raise LeapUnavailable(
                f"the service is in {got} tracking mode, not {want}.\n"
                "  Every recording would be in the wrong frame, so this is "
                "fatal.\n"
                "  Next step: set the mode in the Ultraleap Control Panel, or "
                f"run with --mode {str(got).split('.')[-1].lower()} if that is "
                "what you meant, then:\n"
                f"  {CHECK_SCRIPT}"
            )
        self.tracking_mode_confirmed = True
        log.info("tracking mode confirmed: %s", got)

    def _safe_disconnect(self) -> None:
        """Close the LeapC connection, whatever state it is in."""
        if self._connection is not None:
            try:
                self._connection.disconnect()
            except Exception as e:               # pragma: no cover - hardware path
                log.debug("disconnect complained (ignored): %s", e)
            self._connection = None
        self._listener = None
        self.connected = False

    def stop(self) -> None:
        """Disconnect. Safe to call twice, and safe if `start()` failed."""
        self._safe_disconnect()

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
