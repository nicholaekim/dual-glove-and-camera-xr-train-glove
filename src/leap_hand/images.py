"""The raw IR images: what the camera actually saw, not what it solved.

An `.lmt` recording and our JSONL both hold *solved skeletons*. Neither can
answer the one question the Phase 2 gate exists to answer — does the IR
camera see a hand inside the black StretchSense glove? — because when the
tracker reports nothing there is nothing in either file. The brightness
images are the evidence: a photograph of the glove under 850 nm light, with
the tracker's verdict written next to it.

Three pieces, and they are deliberately small:

  `enable_images(connection)`   turns the image stream on and **confirms** it.
      The flag is `leap.enums.PolicyFlag.Images` — `leap.PolicyFlag` is an
      AttributeError, the bindings do not re-export it. `set_policy_flags`
      waits for the Policy event and returns the flags now in force, so the
      confirmation is free: if `Images` is not in that list the Control Panel
      has "Allow Images" off and every snapshot would be blank.

  `image_to_numpy(image)`       one `leap.Image` -> an 8-bit numpy array.
      The wrapper exposes only `matrix_version`, so the pixels come from
      `image.c_data`: `.properties.width/.height/.bpp`, `.data`, `.offset`.
      **The copy is not optional.** LeapC hands out a pointer into a buffer it
      reuses for the next frame; a view would be quietly overwritten between
      the callback and the PNG write.

  `ImageSampler`                a listener that keeps the newest image pair.
      Not a `leap.Listener` subclass: this module must import on a machine
      with no bindings (every script here runs with `--mock`). It carries the
      callback methods and `attach(connection)` wraps it in a real Listener.

Images are ~90 Hz of full frames, far more than any snapshot tool wants, so
the sampler keeps only the most recent pair and counts the rest as skipped.

Units and layout: `bpp` is 1 for the IR brightness images, one plane per eye,
so an array is (height, width) uint8 — `cv2.imwrite` writes that as an 8-bit
greyscale PNG with no conversion. A hypothetical bpp > 1 becomes
(height, width, bpp) and is passed through untouched.
"""
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .stream import LeapStream, LeapUnavailable

log = logging.getLogger(__name__)

# Left eye first, matching event.image[0] / event.image[1] and the _L / _R
# suffix every snapshot filename carries.
SIDES = ("L", "R")

# What a mock pair pretends to be. The real Stereo IR 170 reports its own size
# (read it from a sidecar JSON of a real run); this only has to be plausible
# and cheap.
MOCK_SIZE = (384, 384)          # (width, height)


@dataclass
class ImagePair:
    """One stereo pair of IR brightness images, already copied out of LeapC."""

    frame_id: int
    timestamp_us: int            # LeapC clock, same epoch as tracking events
    wall_time: float             # time.time() in the callback: the shared clock
    width: int
    height: int
    bpp: int
    left: np.ndarray
    right: np.ndarray

    def sides(self) -> List[Tuple[str, np.ndarray]]:
        """[("L", left), ("R", right)] — the order snapshots are written in."""
        return [("L", self.left), ("R", self.right)]

    @property
    def size_text(self) -> str:
        return f"{self.width}x{self.height}x{self.bpp}"


# --- policy ----------------------------------------------------------------
def enable_images(connection) -> list:
    """Turn the IR image stream on and confirm the service agreed.

    Args:
        connection: a live `leap.Connection` (`LeapStream.connection`).

    Returns:
        The policy flags now in force, as the bindings report them.

    Raises:
        LeapUnavailable: images could not be enabled — almost always "Allow
            Images" being off in the Ultraleap Control Panel.
    """
    leap = LeapStream.import_leap()
    try:
        flags = connection.set_policy_flags(
            flags_to_set=[leap.enums.PolicyFlag.Images])
    except Exception as e:
        raise LeapUnavailable(
            f"could not enable the IR image policy ({type(e).__name__}: {e}).\n"
            "  Next step: open the Ultraleap Control Panel and turn 'Allow "
            "Images' on, then re-run.\n"
            "  No camera to hand? Every script here takes --mock."
        ) from e

    names = {getattr(f, "name", str(f)) for f in (flags or [])}
    if "Images" not in names:
        raise LeapUnavailable(
            "the tracking service refused the Images policy (flags now in "
            f"force: {', '.join(sorted(names)) or 'none'}).\n"
            "  Without it every snapshot would be blank, so this is fatal.\n"
            "  Next step: open the Ultraleap Control Panel, turn 'Allow "
            "Images' on, then re-run."
        )
    log.info("image policy active: %s", ", ".join(sorted(names)))
    return flags


# --- pixels ----------------------------------------------------------------
def _raw_bytes(data, offset: int, nbytes: int):
    """`nbytes` of image data at `offset`, from cffi CData or any buffer.

    The SDK hands over a `void *`; the tests hand over a real
    `ffi.new("uint8_t[]", ...)` or a plain `bytes`, and both must work so the
    conversion is testable without a camera in front of it.
    """
    try:
        from leapc_cffi import ffi          # noqa: PLC0415 — optional dependency
    except ImportError:                     # no bindings: buffer protocol only
        ffi = None

    if ffi is not None and isinstance(data, ffi.CData):
        ptr = ffi.cast("uint8_t *", data) + offset
        return ffi.buffer(ptr, nbytes)
    view = memoryview(data).cast("B")
    if len(view) < offset + nbytes:
        raise ValueError(
            f"image buffer holds {len(view)} bytes, need {offset + nbytes}")
    return view[offset:offset + nbytes]


def image_to_numpy(image) -> np.ndarray:
    """One `leap.Image` -> an 8-bit array, copied out of LeapC's own buffer.

    Args:
        image: a `leap.Image` (or anything exposing the same `c_data`, which
            is what the tests pass).

    Returns:
        (height, width) uint8 for the IR brightness images (bpp 1), or
        (height, width, bpp) for anything wider.

    Raises:
        ValueError: the image properties are empty — no frame behind them.
    """
    c = getattr(image, "c_data", image)
    props = c.properties
    width, height, bpp = int(props.width), int(props.height), int(props.bpp)
    if width <= 0 or height <= 0 or bpp <= 0:
        raise ValueError(
            f"empty image properties (width {width}, height {height}, "
            f"bpp {bpp}) — the frame carries no pixels")

    nbytes = width * height * bpp
    offset = int(getattr(c, "offset", 0) or 0)
    # .copy() is the whole point: LeapC reuses this buffer for the next frame.
    flat = np.frombuffer(_raw_bytes(c.data, offset, nbytes), np.uint8).copy()
    return flat.reshape((height, width) if bpp == 1 else (height, width, bpp))


# --- the listener ----------------------------------------------------------
class ImageSampler:
    """Keeps the newest IR image pair off the SDK's polling thread.

    Deliberately not a `leap.Listener` subclass — see the module docstring.
    `attach(connection)` builds the subclass and registers it; `latest()`
    takes the pair the main thread has not seen yet.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._pair: Optional[ImagePair] = None
        self._listener = None
        self.received = 0        # image events that produced a pair
        self.skipped = 0         # pairs overwritten before anyone took them
        self.errors = 0          # image events that could not be converted
        self.last_error: Optional[str] = None

    # --- callbacks (the names leap.Listener dispatches on) --------------
    def on_image_event(self, event) -> None:
        try:
            images = event.image
            left = image_to_numpy(images[0])
            right = image_to_numpy(images[1])
            props = images[0].c_data.properties
            info = event.c_data.info
            pair = ImagePair(
                frame_id=int(info.frame_id),
                timestamp_us=int(info.timestamp),
                wall_time=time.time(),
                width=int(props.width),
                height=int(props.height),
                bpp=int(props.bpp),
                left=left,
                right=right,
            )
        except Exception as e:                  # pragma: no cover - hardware path
            self.errors += 1
            self.last_error = f"{type(e).__name__}: {e}"
            log.warning("could not read an image event: %s", self.last_error)
            return

        with self._lock:
            if self._pair is not None:
                self.skipped += 1
            self._pair = pair
            self.received += 1

    # --- consumer side ---------------------------------------------------
    def latest(self, clear: bool = True) -> Optional[ImagePair]:
        """The newest pair nobody has taken yet, or None."""
        with self._lock:
            pair = self._pair
            if clear:
                self._pair = None
            return pair

    def wait_for_pair(self, timeout: float = 2.0,
                      poll: float = 0.01) -> Optional[ImagePair]:
        """Block up to `timeout` seconds for the next pair. None on timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            pair = self.latest()
            if pair is not None:
                return pair
            time.sleep(poll)
        return None

    # --- wiring ----------------------------------------------------------
    def attach(self, connection) -> None:
        """Enable the image policy and start receiving on `connection`."""
        leap = LeapStream.import_leap()
        enable_images(connection)
        sampler = self

        class _ImageListener(leap.Listener):
            def on_image_event(self, event):
                sampler.on_image_event(event)

        self._listener = _ImageListener()
        connection.add_listener(self._listener)

    def detach(self, connection) -> None:
        """Stop receiving. Safe without a matching `attach`, or after stop()."""
        if self._listener is None or connection is None:
            self._listener = None
            return
        try:
            connection.remove_listener(self._listener)
        except Exception as e:                  # pragma: no cover - hardware path
            log.debug("removing the image listener complained (ignored): %s", e)
        self._listener = None


class MockImageSampler:
    """Synthetic gradient pairs, behind the `ImageSampler` API.

    The pipeline — sampler to numpy to `cv2.imwrite` to a JSON sidecar — is
    the part worth testing without hardware, so the "pixels" only have to be
    an 8-bit image that changes from snapshot to snapshot. A diagonal ramp
    plus a moving bright band does that and is obviously synthetic on sight,
    which matters: nobody must ever mistake one of these for evidence.
    """

    def __init__(self, size: Tuple[int, int] = MOCK_SIZE, hz: float = 90.0):
        self.width, self.height = int(size[0]), int(size[1])
        self.hz = float(hz)
        self.received = 0
        self.skipped = 0
        self.errors = 0
        self.last_error: Optional[str] = None
        self._i = 0
        self._ramp = np.add.outer(
            np.linspace(0, 120, self.height, dtype=np.float32),
            np.linspace(0, 120, self.width, dtype=np.float32),
        )

    def _eye(self, phase: float, bias: int) -> np.ndarray:
        band = 40.0 * np.cos(
            2.0 * np.pi * (np.linspace(0.0, 1.0, self.width, dtype=np.float32)
                           + phase))
        img = self._ramp + band[None, :] + float(bias)
        return np.clip(img, 0, 255).astype(np.uint8)

    def latest(self, clear: bool = True) -> ImagePair:
        i = self._i
        if clear:
            self._i += 1
        phase = (i % 16) / 16.0
        now = time.time()
        self.received += 1
        return ImagePair(
            frame_id=i,
            # Wall clock in microseconds, which is also what MockLeapStream
            # stamps its hands with, so the image/tracking pairing a sidecar
            # reports is a sane small number in mock runs too.
            timestamp_us=int(now * 1e6),
            wall_time=now,
            width=self.width,
            height=self.height,
            bpp=1,
            left=self._eye(phase, bias=0),
            right=self._eye(phase, bias=12),   # the eyes must not be identical
        )

    def wait_for_pair(self, timeout: float = 2.0,
                      poll: float = 0.01) -> ImagePair:
        return self.latest()

    def attach(self, connection) -> None:
        """No-op: there is no connection behind a mock."""

    def detach(self, connection) -> None:
        """No-op."""


def open_sampler(stream, mock: bool = False):
    """The one line the snapshot tools use: a real sampler, or the mock.

    Args:
        stream: a started `LeapStream` (ignored when `mock`).
        mock: return a `MockImageSampler` instead of touching the SDK.

    Raises:
        LeapUnavailable: no live connection behind `stream`, or the Images
            policy was refused.
    """
    if mock:
        return MockImageSampler()
    connection = getattr(stream, "connection", None)
    if connection is None:
        raise LeapUnavailable(
            "IR snapshots need a live LeapStream (there is no connection "
            "behind a mock stream).\n"
            "  Next step: drop --mock, or run on a machine with the camera "
            "attached."
        )
    sampler = ImageSampler()
    sampler.attach(connection)
    return sampler


# --- writing ---------------------------------------------------------------
@dataclass
class Snapshot:
    """One written snapshot: the two PNGs, the sidecar, and what was tracked."""

    index: int
    label: str
    pair: ImagePair
    png_paths: List[str] = field(default_factory=list)
    json_path: str = ""
    hands: List[dict] = field(default_factory=list)
    tracking_dt_ms: Optional[float] = None
    mean_pixel: dict = field(default_factory=dict)

    @property
    def saw_hand(self) -> bool:
        return bool(self.hands)

    @property
    def tracking_text(self) -> str:
        """The one line that makes a photo evidence: what the tracker said."""
        if not self.hands:
            return "no hand tracked"
        parts = [f"{h['hand_side']}(id {h['hand_id']})" for h in self.hands]
        dt = ("" if self.tracking_dt_ms is None
              else f", {self.tracking_dt_ms:+.0f} ms from the image")
        return f"tracked: {', '.join(parts)}{dt}"


class HandTrail:
    """Recent tracked hands, grouped by tracking frame, for pairing with images.

    Tracking events and image events are separate streams on the same LeapC
    clock, so "what did the tracker see in this photo" means "the tracking
    event nearest this image's timestamp". Hands arrive one at a time from
    `LeapStream.drain()`, so they are regrouped by `frame_id` here.

    `PAIR_WINDOW_MS` is what keeps a sidecar honest. Both streams run at about
    90 Hz and the trail is drained either side of every capture, so a genuine
    pairing lands within a few milliseconds. Without the window, a still taken
    while nothing is tracked would be matched to whatever hand was last seen —
    possibly minutes and a whole condition ago — and the photograph would
    claim a hand that was not there, which is the one mistake this file exists
    to prevent.
    """

    # Nearest tracking event further away than this counts as "no hand".
    PAIR_WINDOW_MS = 100.0

    def __init__(self, maxlen: int = 900):
        # [(timestamp_us, frame_id, [hand dicts])], oldest first
        self._frames = deque(maxlen=maxlen)

    def clear(self) -> None:
        """Forget everything — call between conditions of an experiment."""
        self._frames.clear()

    def add(self, lh) -> None:
        entry = {
            "hand_side": lh.hand_side,
            "hand_id": int(lh.hand_id),
            "palm_pos": [round(float(c), 6) for c in lh.palm_pos],
            "visible_time_us": int(lh.visible_time_us),
        }
        if self._frames and self._frames[-1][1] == lh.frame_id:
            self._frames[-1][2].append(entry)
            return
        self._frames.append((int(lh.timestamp_us), int(lh.frame_id), [entry]))

    def nearest(self, timestamp_us: int, window_ms: Optional[float] = None):
        """(hands, dt_ms) for the tracking event nearest `timestamp_us`.

        `dt_ms` is the tracking event's time minus the image's, so a positive
        value means the tracker's frame is the later of the two. Returns
        `([], None)` when no hand has been seen, or when the nearest one is
        further away than `window_ms` (default `PAIR_WINDOW_MS`) — a stale
        hand is not evidence about this photograph.
        """
        if not self._frames:
            return [], None
        limit = self.PAIR_WINDOW_MS if window_ms is None else window_ms
        ts, _fid, hands = min(self._frames,
                              key=lambda f: abs(f[0] - timestamp_us))
        dt_ms = (ts - timestamp_us) / 1000.0
        if abs(dt_ms) > limit:
            return [], None
        return list(hands), dt_ms


def write_snapshot(out_dir, label: str, index: int, pair: ImagePair,
                   hands: Optional[List[dict]] = None,
                   tracking_dt_ms: Optional[float] = None,
                   note: str = "") -> Snapshot:
    """Write `<label>_<k>_L.png`, `_R.png` and one JSON sidecar.

    The sidecar is what turns a photo into evidence: it states the frame id,
    both clocks, the image size and — the point of the whole exercise —
    whether the tracker reported a hand at that instant, so a still of a
    gloved hand can never be filed without its verdict.

    Args:
        out_dir: folder to write into (created if missing).
        label: condition or run label, e.g. "glove_liner".
        index: snapshot number within the run.
        pair: the images to write.
        hands: hands from the nearest tracking event (see `HandTrail`).
        tracking_dt_ms: how far that tracking event sat from the image.
        note: free text copied into the sidecar (mock runs say so here).

    Returns:
        The `Snapshot` record, with the paths actually written.
    """
    import cv2                              # noqa: PLC0415 — heavy import

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_{index:03d}"
    hands = list(hands or [])

    snap = Snapshot(index=index, label=label, pair=pair, hands=hands,
                    tracking_dt_ms=tracking_dt_ms)
    for side, array in pair.sides():
        path = out_dir / f"{stem}_{side}.png"
        if not cv2.imwrite(str(path), array):
            raise OSError(f"cv2 could not write {path}")
        snap.png_paths.append(str(path))
        snap.mean_pixel[side] = round(float(array.mean()), 2)

    sidecar = out_dir / f"{stem}.json"
    payload = {
        "label": label,
        "index": index,
        "frame_id": pair.frame_id,
        "timestamp_us": pair.timestamp_us,
        "wall_time": pair.wall_time,
        "width": pair.width,
        "height": pair.height,
        "bpp": pair.bpp,
        "units": "m",
        "space": "leap_desktop",
        "images": [Path(p).name for p in snap.png_paths],
        "mean_pixel": snap.mean_pixel,
        "hands": hands,
        "hand_count": len(hands),
        "tracking_dt_ms": (None if tracking_dt_ms is None
                           else round(tracking_dt_ms, 3)),
        "tracking": snap.tracking_text,
    }
    if note:
        payload["note"] = note
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    snap.json_path = str(sidecar)
    return snap
