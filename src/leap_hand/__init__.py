"""leap_hand: Ultraleap Stereo IR 170 pipeline (LeapC -> 26 OpenXR joints).

The third sensor in this repo, shaped like the other two:

  src/xr_hand    StretchSense glove   (OSC -> 26 joints, parent-relative)
  src/cam_hand   webcam + MediaPipe   (21 landmarks)
  src/leap_hand  Ultraleap IR 170     (LeapC -> 26 joints, parent-relative)

The camera measures what the glove cannot: where the hand is, how it is
oriented, finger spread, thumb opposition, wrist angle. Its output is
converted into the glove's own `HandFrame` convention (parent-relative
translations, XYZW quaternions, metres), so `playback.py`,
`export_keypoints21.py`, `export_prof_format.py` and the rest read camera
recordings with no changes at all.

Nothing here imports `leap` at module level: the SDK and its Python bindings
are installed by hand (see `scripts/leap/setup_bindings.ps1`), and every
script in this repo must still run — with `--mock` — on a machine that has
neither.
"""
__version__ = "0.1.0"
