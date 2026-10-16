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

Tracking hints (`hints=("hand_on_object",)`). LeapC 6.2 lets a client tell
the service how the hands will be used, per device, with
`LeapSetDeviceHints(connection, device, hints[])`; `hand_on_object` asks for
the tracking Ultraleap tunes for hands occluded by a held object, and it
needs the service's Hyperion license. The bindings wrap neither call, so
they go through `leapc_cffi` directly (`_leapc()`). Three facts from
LeapC.h decide how:

  * a hint the service does not know or is not licensed for is ignored
    without an error, so the Hyperion flag is read first
    (`LeapCheckLicenseFlag`) and its absence is fatal: a take labelled
    "hand_on_object" that was tracked without it is worse than no take;
  * the hints "will only be interpreted for multi-device aware clients" and
    the call "will return an error if the client is not", so a stream with
    hints opens its connection multi-device aware, and then subscribes to
    each device's events, which such a client must do to receive them;
  * the call takes an OPEN device handle (`LeapOpenDevice`), and the
    bindings' `connection.get_devices()` returns unopened devices whose
    `c_data_device` is None, so each device is opened for the call.

With no hints nothing of this runs and the connection is exactly the one it
always was. `hints_requested`, `hints_confirmed` and `license_hyperion` say
what happened; `hint_metadata()` is how the recorders write them down.
"""
import logging
import time
from contextlib import ExitStack
from queue import Empty, Full, Queue
from typing import List, Optional, Sequence, Tuple

from .to_openxr import leap_hand_from_api
from .types import LeapHand

log = logging.getLogger(__name__)

# Same shape as xr_hand.receiver.QueueItem: (hand_side, payload).
QueueItem = Tuple[str, LeapHand]

SETUP_SCRIPT = r"powershell -ExecutionPolicy Bypass -File scripts\leap\setup_bindings.ps1"
CHECK_SCRIPT = r"python scripts\leap\check_setup.py"

TRACKING_MODES = ("desktop", "hmd", "screentop")

# The tracking hints a recording may ask for (LeapC.h, \ingroup Hints). Only
# the one this repo has a use and a test for: the service ignores a name it
# does not know without saying so, so a typo must stop here instead.
KNOWN_HINTS = ("hand_on_object",)
# The license feature flag the hand_on_object hint needs (LeapC.h).
HYPERION_FLAG = "Hyperion"
LEAP_SUCCESS = 0                       # eLeapRS_Success


def _leapc():
    """`(ffi, libleapc)` of the compiled LeapC bindings, imported on use.

    The one way this module reaches LeapC functions the `leap` package does
    not wrap. A function of its own so the tests can replace it: the fake
    SDK objects have no cffi behind them.
    """
    from leapc_cffi import ffi, libleapc  # noqa: PLC0415 — optional, lazy
    return ffi, libleapc


def check_hints(hints: Sequence[str]) -> Tuple[str, ...]:
    """`hints` as a tuple, each name once, or ValueError naming the bad one."""
    if isinstance(hints, (str, bytes)):
        raise TypeError("hints is a sequence of hint names, not one string: "
                        f"hints=({hints!r},)")
    out: List[str] = []
    for hint in hints:
        if hint not in KNOWN_HINTS:
            raise ValueError(f"unknown tracking hint {hint!r}; the known ones "
                             f"are {KNOWN_HINTS}")
        if hint not in out:
            out.append(hint)
    return tuple(out)


def hint_metadata(source) -> dict:
    """The three keys a take's metadata carries about tracking hints.

      tracking_hints     the hints the stream was asked for, [] for none
      hints_confirmed    True only when the service accepted them on every
                         device (a mock says True when it was given any)
      license_hyperion   the service's Hyperion flag as read, None when it
                         was not read (no hints asked for, or no stream)

    Read off whatever `source` is (a `LeapStream`, the mock, a test double
    or None), so a source that knows nothing about hints reads as none.
    """
    license_flag = getattr(source, "license_hyperion", None)
    return {
        "tracking_hints": list(getattr(source, "hints_requested", None) or []),
        "hints_confirmed": bool(getattr(source, "hints_confirmed", False)),
        "license_hyperion": None if license_flag is None else bool(license_flag),
    }


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
        hints: tracking hints to request on every device, from
            `KNOWN_HINTS` (ValueError otherwise). Empty: none, and nothing
            hint-related is called. See the module docstring.
    """

    def __init__(
        self,
        mode: str = "desktop",
        device_timeout: float = 5.0,
        mode_timeout: float = 5.0,
        queue_maxsize: int = 480,
        hints: Sequence[str] = (),
    ):
        if mode not in TRACKING_MODES:
            raise ValueError(f"mode must be one of {TRACKING_MODES}, got {mode!r}")
        self.mode = mode
        self.device_timeout = float(device_timeout)
        self.mode_timeout = float(mode_timeout)
        self.queue: "Queue[QueueItem]" = Queue(maxsize=queue_maxsize)
        # the hints asked for, checked now so a bad name fails before any
        # connection is made
        self.hints_requested: List[str] = list(check_hints(hints))

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
        # False until the service has accepted `hints_requested` on every
        # device; the Hyperion flag as read, None until it is read (it is
        # read only when hints are asked for). Kept after stop(): a recorder
        # writes them down after the stream is closed.
        self.hints_confirmed = False
        self.license_hyperion: Optional[bool] = None

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

        With hints, the device wait is followed by the Hyperion check and the
        hints on every device (`_apply_hints`), then the mode is verified
        and the queue emptied once more, so no frame from before the hint
        took effect is counted either. The hints come before the mode
        verification, not after it: a multi-device aware connection has to
        subscribe to a device before that device's events reach it, and the
        verification waits for one of those events.

        Raises:
            LeapUnavailable: bindings missing, service not running, no device
                within `device_timeout` seconds, the tracking mode could not
                be set and confirmed, or hints were asked for and the
                service has no Hyperion license or refused them. Whatever the
                reason, the LeapC connection is closed before the error
                leaves this method.
        """
        self._leap = self.import_leap()
        self._warned_device = False
        self.hints_confirmed = False
        self.license_hyperion = None
        # response_timeout bounds the mode round trip below; the default of
        # 10 s is a long time to sit in front of a camera wondering.
        options = {"response_timeout": self.mode_timeout}
        if self.hints_requested:
            # LeapC interprets hints only from a multi-device aware client.
            options["multi_device_aware"] = True
        self._connection = self._leap.Connection(**options)
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
            self._apply_hints()
            self._verify_tracking_mode()
            # Frames that arrived between the set and its confirmation were
            # queued in an unconfirmed mode, and with hints, before the hint
            # took effect; only what follows counts.
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

    # --- tracking hints ---------------------------------------------------
    def _apply_hints(self) -> None:
        """Read the Hyperion flag, then set `hints_requested` on every device.

        Nothing is called without hints. Every failure is fatal, for the
        same reason a wrong tracking mode is: the recording would carry a
        condition it was not made under.
        """
        if not self.hints_requested:
            return
        ffi, libleapc = self._load_leapc()
        conn = self._connection.get_connection_ptr()
        self._check_hyperion(ffi, libleapc, conn)

        try:
            devices = list(self._connection.get_devices())
        except Exception as e:
            raise LeapUnavailable(
                f"could not list the devices to set the "
                f"{self._hint_text()} hint on ({type(e).__name__}: {e}).\n"
                f"  Next step: {CHECK_SCRIPT}, or record without --hint"
            ) from e
        if not devices:
            raise LeapUnavailable(
                f"the service lists no device to set the {self._hint_text()} "
                "hint on, although it reported one a moment ago.\n"
                "  Next step: check the cable and the Ultraleap Control "
                f"Panel, then: {CHECK_SCRIPT}")

        # One C array for every device. `strings` owns the bytes the array
        # points at and stays referenced until the last call has returned.
        strings = [ffi.new("char[]", h.encode("ascii"))
                   for h in self.hints_requested]
        array = ffi.new("const char*[]", strings + [ffi.NULL])
        for device in devices:
            self._hint_device(libleapc, conn, device, array)
        self.hints_confirmed = True
        log.info("tracking hints %s set on %d device(s), serial %s",
                 self.hints_requested, len(devices), self.device_serial)

    def _load_leapc(self):
        try:
            return _leapc()
        except ImportError as e:
            raise LeapUnavailable(
                "the compiled LeapC bindings (leapc_cffi) are not importable "
                f"here ({e}), and tracking hints are set through them.\n"
                f"  Next step: {SETUP_SCRIPT}, then {CHECK_SCRIPT}, or record "
                "without --hint"
            ) from e

    def _check_hyperion(self, ffi, libleapc, conn) -> None:
        """`license_hyperion` from the service, or LeapUnavailable."""
        enabled = ffi.new("bool*")
        code = libleapc.LeapCheckLicenseFlag(conn, HYPERION_FLAG.encode("ascii"),
                                             enabled)
        if code != LEAP_SUCCESS:
            raise LeapUnavailable(
                "the Ultraleap service did not answer the Hyperion license "
                f"check (LeapCheckLicenseFlag returned {self._rs_text(code)}), "
                f"so whether the {self._hint_text()} hint would be honoured is "
                "unknown and the recording could be labelled with a condition "
                "it was not made under.\n"
                f"  Next step: run {CHECK_SCRIPT}, or record without --hint")
        self.license_hyperion = bool(enabled[0])
        if not self.license_hyperion:
            raise LeapUnavailable(
                "the Ultraleap service here has no Hyperion license, so the "
                f"{self._hint_text()} hint would be ignored and the recording "
                "would be labelled with a condition it was not made under.\n"
                f"  Next step: run {CHECK_SCRIPT}, or record without --hint")

    def _hint_device(self, libleapc, conn, device, array) -> None:
        """Subscribe to one device and set the hints on it, opening it if
        it is not open. A device the bindings list is not open: its
        `c_data_device` is None until `device.open()` gives it a handle."""
        if getattr(device, "c_data_device", None) is not None:
            self._subscribe_and_hint(libleapc, conn, device, array)
            return
        with ExitStack() as opened:
            try:
                opened.enter_context(device.open())
            except Exception as e:
                raise LeapUnavailable(
                    f"could not open the device to set the {self._hint_text()} "
                    f"hint (is another app using the camera?): {e}.\n"
                    "  Next step: close the other Ultraleap apps, then: "
                    f"{CHECK_SCRIPT}"
                ) from e
            # closed again on the way out, as the bindings' own
            # multi-device example does after subscribing
            self._subscribe_and_hint(libleapc, conn, device, array)

    def _subscribe_and_hint(self, libleapc, conn, device, array) -> None:
        try:
            self._connection.subscribe_events(device)
        except Exception as e:
            raise LeapUnavailable(
                "could not subscribe to the device's events, which a "
                "multi-device aware connection (the kind hints need) must do "
                f"to receive them ({type(e).__name__}: {e}).\n"
                f"  Next step: {CHECK_SCRIPT}, or record without --hint"
            ) from e
        code = libleapc.LeapSetDeviceHints(conn, device.c_data_device, array)
        if code != LEAP_SUCCESS:
            raise LeapUnavailable(
                f"the service refused the {self._hint_text()} hint "
                f"(LeapSetDeviceHints returned {self._rs_text(code)}, device "
                f"{self.device_serial}), so the recording would be labelled "
                "with a condition it was not made under.\n"
                f"  Next step: run {CHECK_SCRIPT}, or record without --hint")

    def _hint_text(self) -> str:
        return ", ".join(self.hints_requested)

    def _rs_text(self, code) -> str:
        """An eLeapRS code as its name and hex, e.g. 'Timeout (0xE2010004)'."""
        try:
            value = int(code)
        except (TypeError, ValueError):
            return repr(code)
        name = None
        try:
            name = self._leap.enums.RS(value).name
        except Exception:
            pass
        text = f"0x{value & 0xFFFFFFFF:08X}"
        return f"{name} ({text})" if name else text

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
    hints: Sequence[str] = (),
):
    """The one line every script uses: a real stream, or the mock.

    Returns an already-started source with `drain()`, `stop()` and the stats
    attributes the HUDs read. `pose` is honoured by the mock only; `hints`
    by both (the mock only records them, see `MockLeapStream`).
    """
    if mock:
        from .mock import MockLeapStream
        source = MockLeapStream(pose=pose, hints=hints)
    else:
        source = LeapStream(mode=mode, device_timeout=device_timeout,
                            hints=hints)
    source.start()
    return source
