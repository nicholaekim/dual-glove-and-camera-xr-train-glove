"""Read and write LeapC's own `.lmt` recordings.

The JSONL files this repo writes are the deliverable: they carry the 26
joints in the glove's convention and every existing tool reads them. An
`.lmt` is the raw tracking stream as LeapC recorded it — no conversion, no
throttling, nothing of ours in the middle. It is worth capturing next to a
session because it is the only artefact that can be re-processed if the joint
mapping or the unit handling later turns out to be wrong. On hardware day
that is not a hypothetical.

  `RawRecording` attaches `leap.Recorder` to a live `LeapStream` for the
  duration of a take, writing `<take>.lmt` beside the JSONL.

  `iter_recording(path)` reads one back, yielding `LeapHand` objects through
  the same `leap_hand_from_api` conversion the live path uses. Replay does
  not go through `Connection` listeners: the events are fed in directly.

Both need the `leap` bindings; neither is on the mock path.
"""
from pathlib import Path
from typing import Iterator

from .stream import LeapStream, LeapUnavailable
from .to_openxr import leap_hand_from_api
from .types import LeapHand


def iter_recording(path: str | Path) -> Iterator[LeapHand]:
    """Yield every hand of every tracking event in a `.lmt` recording.

    Raises:
        LeapUnavailable: the bindings are not installed (the message says how).
        FileNotFoundError: no such recording.
    """
    leap = LeapStream.import_leap()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such Leap recording: {path}")

    with leap.Recording(str(path), mode="r") as recording:
        for event in recording:
            hands = getattr(event, "hands", None)
            if not hands:
                continue        # non-tracking events in the stream
            for hand in hands:
                yield leap_hand_from_api(hand, event, frame_age_us=None)


class RawRecording:
    """Write a `.lmt` of everything a live `LeapStream` sees, as a context.

    `leap.Recording` insists on being used as a context manager (it opens the
    file through LeapRecordingOpen), and `leap.Recorder` is a Listener that
    writes each tracking event into it, so this just owns both for the life
    of a take:

        with RawRecording(stream, path):
            ...  # record the take as usual
    """

    def __init__(self, stream: LeapStream, path: str | Path):
        self.stream = stream
        self.path = Path(path)
        self._recording_cm = None
        self._recorder = None

    def __enter__(self) -> "RawRecording":
        leap = LeapStream.import_leap()
        connection = getattr(self.stream, "connection", None)
        if connection is None:
            raise LeapUnavailable(
                "raw .lmt capture needs a live LeapStream (there is no "
                "connection behind a mock stream).\n"
                "  Next step: drop --raw, or run without --mock on a machine "
                "with the camera attached."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._recording_cm = leap.Recording(str(self.path), mode="w")
        recording = self._recording_cm.__enter__()
        self._recorder = leap.Recorder(recording, auto_start=True)
        connection.add_listener(self._recorder)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        connection = getattr(self.stream, "connection", None)
        if connection is not None and self._recorder is not None:
            try:
                connection.remove_listener(self._recorder)
            except Exception:       # pragma: no cover - hardware path
                pass                # a listener that will not detach is not
                                    # worth failing a finished take over
        if self._recording_cm is not None:
            self._recording_cm.__exit__(exc_type, exc, tb)
        self._recording_cm = None
        self._recorder = None
