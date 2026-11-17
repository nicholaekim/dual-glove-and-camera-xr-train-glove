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
(the then 20-page `figures.pdf` of the results folder, one page per recorded take; the hand-in's
`figures.pdf` holds only the ten accepted takes), was shown a 20-row per-take reading with a reason per row, and confirmed it
in one reply ("Confirm all 20"). The assistant then entered the decisions into
`visual_checks.csv`. The assistant's reading is not an independent visual verification.
The operator reviewed the completed sheet (sent as a file) and approved it in chat ("Approved as filled") at 2026-11-06T00:09:19-05:00; all 20 decisions and their recorded reasons stand as the operator's.

Second confirmation, 2026-11-06T10:49:01-05:00: after the redo (see "Redo of 2026-11-06"
below) the operator looked at the figure.png of the six new takes and of the three
challenged takes, was shown a nine-row reading with a reason per row, and confirmed it in
one reply ("Confirm all"); the assistant entered the six new decisions and revised three
earlier ones (crossed_fingers take 1, thumb_opposition take 1, pinch take 2, each yes to
no). A revised row keeps its original decision and time in its note ("original decision
yes at 2026-11-05T23:52:14-05:00"); only `intended_pose_seen`, `decided_at` and `note`
changed. These nine decisions are the operator's; decision_by operator, entered_by
assistant, as before.

### Selection and exclusions

26 takes were recorded and accepted by the recorder: 20 on 2026-11-05 and 6 in the redo of
2026-11-06 (takes 3 and 4 of crossed_fingers, thumb_opposition and pinch). The recorder
refused 5 further attempts, all on 2026-11-05: open_palm take 1 attempt 1 (left hand lost
for 0.5 s at 18.1 cm); bent_wrist take 1 attempts 1 and 2, bent_wrist take 2 attempt 2 and
peace take 2 attempt 1 (the tracker let go of the hand and re-acquired it).

5 takes were "moving" at the photo's moment: crossed_fingers take 2 (camera); finger_spread
take 2 and pinch take 1 (both sensors); thumb_opposition take 3 and pinch take 3 (camera).

6 takes are a visual no: thumb_opposition take 2 (the photo shows the pinky up and the
thumb out, a different hand shape) and crossed_fingers take 2 (the photo shows curled
fingers while the glove reads an open hand, a transition); crossed_fingers take 1,
thumb_opposition take 1 and pinch take 2 (revised from yes on 2026-11-06, reasons under
"Redo of 2026-11-06"); thumb_opposition take 3 (open hand, thumb pointing away from the
hand).

Selection is the deterministic first-passing-take rule: each pose's takes are walked in
ascending take number and the first take that passes every check is accepted. Selected:
take 1 for open_palm, fist, index_point, thumbs_up, peace, finger_spread and bent_wrist;
crossed_fingers take 3; thumb_opposition take 4; pinch take 4. crossed_fingers take 3,
thumb_opposition take 4 and pinch take 4 were selected because every earlier take of the
pose failed a check (stillness or the visual check), not because they produced a smaller
disagreement.

All takes, the refused attempts, the still sidecars, the full frames and the logs are kept
in the internal archive. The uncropped frames are excluded from the hand-in.

### Redo of 2026-11-06

An independent inspection of the hand-in files (gate G26 in `GATES_paired_pose.md`)
challenged three photos: crossed_fingers take 1 (a narrow V, no visible crossing),
thumb_opposition take 1 (the thumb across a closed hand) and pinch take 2 (no visible
thumb-index contact). Six takes were then recorded on 2026-11-06 from 10:43 to 10:46, gloves
on and the camera running, under a new calibration: takes 3 and 4 of crossed_fingers,
thumb_opposition and pinch. During the redo the recorder's window showed the target pose
(`reference_poses/`): the professor's frame for eight of the ten poses, and a text card with
the pose's definition for crossed_fingers and thumb_opposition, because her set has no
photo of either pose.

The six new decisions (operator, by eye from figure.png, visual criterion as above):

- crossed_fingers take 3: yes, middle finger visibly over the index; Leap puts both tips
  together.
- crossed_fingers take 4: yes, same, clearer.
- thumb_opposition take 3: no, open hand, thumb pointing away from the hand.
- thumb_opposition take 4: yes, thumb across the palm, tip on the pinky tip, other fingers
  extended.
- pinch take 3: yes, clear thumb-index ring (the rule rejects this take for stillness).
- pinch take 4: yes, clear thumb-index ring, held.

The three revisions, at 2026-11-06T10:49:01-05:00, each from yes (2026-11-05T23:52:14-05:00)
to no:

- crossed_fingers take 1: the crossing is not visible in the photo (narrow V).
- thumb_opposition take 1: thumb lies over a closed hand, not opposed to the pinky.
- pinch take 2: thumb-index contact not visible in the photo.

Resulting selection: crossed_fingers take 3 (take 1 visual no; take 2 stillness and visual
no); thumb_opposition take 4 (takes 1 and 2 visual no; take 3 stillness and visual no);
pinch take 4 (take 1 stillness; take 2 visual no; take 3 stillness); the other seven poses
unchanged. The reference pictures were added to the recorder after commit 95b9040 and are
not yet committed.

Reviewer's note: the operator's confirmation, on 2026-11-05 and again on 2026-11-06, was a
per-row reading confirmed in one reply, not a separate reply per row.

### How to read the numbers

The medians and maxima are landmark distances between two model-based reconstructions of
the hand; neither sensor is ground truth. Each is a wrist-centred, independently
palm-aligned shape disagreement from an arrival-time matched, held-pose comparison.

The left glove has a transmission delay of about 0.47 s (measured 5 October 2026); its
`capture_time` is packet arrival, and the delay is not corrected.

bent_wrist is a control: the palm-aligned frame removes the wrist angle, so its distances
are not wrist-angle disagreement.

The largest maxima of the selected takes (peace 77.3 mm at the thumb tip, thumb_opposition
55.2 mm at the index fingertip) are not attributed to either sensor.
