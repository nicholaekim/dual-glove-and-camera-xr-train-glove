"""MediaPipe HandLandmarker wrapper -> 21 keypoints per detected hand.

One class (`HandTracker`) hides the mediapipe tasks API so scripts only see
`CamHand` results. Each hand carries two coordinate sets:

  img    21 x [x_px, y_px, z_rel]  image-space pixels (z is MediaPipe's
                                   relative depth, wrist ~ 0, unitless)
  world  21 x [x_m, y_m, z_m]      metric landmarks in metres, origin at the
                                   hand's geometric centre. MediaPipe assumes
                                   an average-sized hand, so treat these as
                                   approximate metric, not ground truth.

Landmark order is the MediaPipe standard (MP21_NAMES below) — identical to
the glove pipeline's `xr_hand.keypoints21.MP21_NAMES`.

Handedness: frames are fed to the model UNMIRRORED so the recorded geometry
is the true hand; mirroring is applied to the preview window only, never to
the data. The label is then used as MediaPipe reports it (swap_handedness
defaults to False).

That default was measured, not assumed. Older MediaPipe docs say handedness
assumes a mirrored selfie image and should be swapped for unmirrored input,
but on the reference tracker dataset (102 frames with an independent device's
own left/right labels) the unswapped label agrees on 9 of the first 12 frames
and the swapped one on 3 — and an alignment test that mirrors the camera
skeleton and refits confirms the unswapped reconstruction has the same
chirality as the tracker's. See scripts/compare_to_tracker.py, which reports
both. Pass swap_handedness=True if a future model version flips this again;
the two-second check with a webcam is to raise your right hand and read the
on-screen label.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

MP21_NAMES: List[str] = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# fingertip landmark index per finger, for extension profiles
FINGERTIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"


@dataclass
class CamHand:
    hand_side: str      # 'left' | 'right' — the physical hand (labels swapped, see module docstring)
    score: float        # handedness confidence from MediaPipe
    img: list           # 21 x [x_px, y_px, z_rel]
    world: list         # 21 x [x_m, y_m, z_m]


class HandTracker:
    """MediaPipe HandLandmarker in 'video' (webcam) or 'image' (stills) mode.

    Video mode requires strictly increasing timestamps in ms — pass
    time.monotonic()-based stamps to detect(). Image mode ignores them.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        num_hands: int = 2,
        running_mode: str = "video",
        swap_handedness: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        gamma: float = 1.0,
        clahe: float = 0.0,
    ):
        import mediapipe as mp

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"hand landmarker model not found: {model_path}\n"
                "Download it (see README) into the models/ folder.")

        self._mp = mp
        self.swap_handedness = swap_handedness
        self.running_mode = running_mode
        self.gamma = gamma
        self.clahe = clahe

        vision = mp.tasks.vision
        mode = {"video": vision.RunningMode.VIDEO,
                "image": vision.RunningMode.IMAGE}[running_mode]
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mode,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr, ts_ms: Optional[int] = None) -> List[CamHand]:
        """Detect hands in a BGR (OpenCV) frame. Returns [] when none found."""
        import cv2

        from .enhance import boost

        # Landmarks come back in pixel coordinates of this image, and boosting
        # does not move pixels, so results still line up with the original
        # frame the caller draws on.
        frame_bgr = boost(frame_bgr, self.gamma, self.clahe)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self.running_mode == "video":
            if ts_ms is None:
                raise ValueError("video mode needs a monotonic ts_ms per frame")
            result = self._landmarker.detect_for_video(image, int(ts_ms))
        else:
            result = self._landmarker.detect(image)

        h, w = frame_bgr.shape[:2]
        hands: List[CamHand] = []
        for handed, lms, wlms in zip(result.handedness,
                                     result.hand_landmarks,
                                     result.hand_world_landmarks):
            label = handed[0].category_name.lower()
            if self.swap_handedness:
                label = "left" if label == "right" else "right"
            hands.append(CamHand(
                hand_side=label,
                score=float(handed[0].score),
                img=[[lm.x * w, lm.y * h, lm.z] for lm in lms],
                world=[[lm.x, lm.y, lm.z] for lm in wlms],
            ))
        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
