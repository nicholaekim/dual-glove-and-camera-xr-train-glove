"""cam_hand: webcam hand-keypoint pipeline (MediaPipe 21-landmark layout).

Camera-side counterpart of the glove pipeline (`xr_hand`, in this same repo
under `src/`). Both speak the same 21-keypoint convention, so recordings from
either source export to the same CSV / professor-format files and can be
compared frame to frame.
"""
__version__ = "0.1.0"
