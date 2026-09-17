Glove pose dataset - StretchSense XR gloves via XR Trainer
Recorded 2026-07-01 (Nicholas Kim)

7 finger poses x 3 takes x ~5 s, both hands, 5 frames/s per hand.

One folder per pose (1_open_palm ... 7_pinch). Each folder contains:
  *.jsonl                  raw recordings, one file per take: 26 OpenXR
                           joints per hand per frame (parent-relative
                           position + XYZW quaternion), with "pose" and
                           "take" labels on every frame
  <pose>_keypoints21.csv   every frame converted to the 21-keypoint hand
                           layout (MediaPipe / Ultralytics standard)
  <pose>_skeleton.csv      one representative frame per hand - the
                           recorded frame closest to the pose average

Top level:
  keypoints21_frames.csv   all poses combined, one row per frame per hand
  keypoints21_summary.csv  all poses combined, one skeleton per pose x hand
  REPORT.txt               fingertip-extension table per pose + pose
                           separability (leave-one-out nearest-centroid:
                           38/42 samples = 90%; 30/30 on the 5 core poses)

Keypoint format: wrist-centred world XYZ in metres, columns named
WRIST, THUMB_CMC/MCP/IP/TIP, then {INDEX,MIDDLE,RING}_FINGER_{MCP,PIP,
DIP,TIP} and PINKY_{MCP,PIP,DIP,TIP} - the standard 21-point order.

Known limitation: the pinch pose is not reliably sensed by the gloves.
Thumb opposition is rotation rather than finger flexion, so the stretch
sensors barely register it (XR Trainer's own hand model shows the same).
Kept here as documentation; relevant to the complementary-sensor work
planned for fall.
