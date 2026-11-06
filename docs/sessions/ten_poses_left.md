### Camera orientation

The protocol specified the Ultraleap IR 170 lying flat, lenses up. The uncropped frames
(`<take>.full.jpg`) show that it was tilted toward the operator: the operator's face and
torso are in frame. No independent measurement of the camera angle was recorded. The
accepted takes passed the tracking, timing and coordinate-frame checks. The results here
must not be described as measurements under the specified flat-camera configuration.

### Visual acceptance criterion

Visual acceptance criterion clarified after recording but before visual decisions were entered; all 20 takes were assessed using the clarified criterion.

Original wording: "did the photo and both models show the intended pose?"

Clarified wording (docs/paired_pose.md step 11): "Yes: the photo shows the intended pose
being held (not a transition, not another pose) and each model shows that same hand as far
as its sensor can represent it. A sensor's known inability is the disagreement under study,
not a no: the glove reads finger flexion only, with a template spread and no wrist, so its
model cannot show a pinch closing, a true spread or crossed fingers, and no model shows the
wrist angle of bent_wrist (the control). No: the photo shows a different pose or a hand in
transition, or a model contradicts the photo beyond its sensor's limits (a glove reading an
open hand while the photo shows a fist)."

Times, 2026-11-05: the sheet (`visual_checks.csv`) was all blank at 23:47; it was filled at
23:52.

Rationale: read literally, the original wording would have sent pinch (the glove cannot
close a pinch), finger_spread (template spread), crossed_fingers and bent_wrist (the
comparison removes the wrist angle) to re-record for what their sensors cannot show, which
no re-recording could change. The clarification applies to every take alike and does not
look at any distance. Recorded as gate G23 in `GATES_paired_pose.md`.

### Decision provenance

decision_by: operator. entered_by: assistant. The operator looked at all 20 figures
(`figures.pdf`), was shown a 20-row per-take reading with a reason per row, and confirmed it
in one reply ("Confirm all 20"). The assistant then entered the decisions into
`visual_checks.csv`. The assistant's reading is not an independent visual verification.
The operator reviewed the completed sheet (sent as a file) and approved it in chat ("Approved as filled") at 2026-11-06T00:09:19-05:00; all 20 decisions and their recorded reasons stand as the operator's.

### Selection and exclusions

20 takes were recorded and accepted by the recorder. The recorder refused 5 further
attempts: open_palm take 1 attempt 1 (left hand lost for 0.5 s at 18.1 cm); bent_wrist take
1 attempts 1 and 2, bent_wrist take 2 attempt 2 and peace take 2 attempt 1 (the tracker let
go of the hand and re-acquired it).

3 takes were "moving" at the photo's moment: crossed_fingers take 2 (camera); finger_spread
take 2 and pinch take 1 (both sensors).

2 takes were a visual no: thumb_opposition take 2 (the photo shows the pinky up and the
thumb out, a different hand shape) and crossed_fingers take 2 (the photo shows curled
fingers while the glove reads an open hand, a transition).

Selection is the deterministic first-passing-take rule: each pose's takes are walked in
ascending take number and the first take that passes every check is accepted. pinch take 2
was selected because pinch take 1 failed the stillness check, not because take 2 produced a
smaller disagreement.

All takes, the refused attempts, the still sidecars, the full frames and the logs are kept
in the internal archive. The uncropped frames are excluded from the hand-in.

### How to read the numbers

The medians and maxima are landmark distances between two model-based reconstructions of
the hand; neither sensor is ground truth. Each is a wrist-centred, independently
palm-aligned shape disagreement from an arrival-time matched, held-pose comparison.

The left glove has a transmission delay of about 0.47 s (measured 5 October 2026); its
`capture_time` is packet arrival, and the delay is not corrected.

bent_wrist is a control: the palm-aligned frame removes the wrist angle, so its distances
are not wrist-angle disagreement.

The largest maxima (peace 77.3 mm at the thumb tip, crossed_fingers 65.6 mm at the thumb
tip) are not attributed to either sensor.
