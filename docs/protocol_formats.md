# Protocol recordings: file formats (normative)

Companion to `grasp_and_flexion_protocol_plan.md`. Three tools produce and
consume these files: `scripts/record_protocol.py` (Sets B and C),
`scripts/leap/record_poses.py --protocol` (Set A), `scripts/check_protocol.py`
and `scripts/package_professor_set.py`. They are built separately, so this
file is the contract. Change it here first, then in the tools.

## 1. Session folder

    recordings/protocol/<set>/<YYYYMMDD_HHMMSS>_<hand>/

`<set>` is `grasps`, `finger_flexion` or `sequences`. `<hand>` is the
operator's hand (`left` or `right`), never the tracker's label.

    session.json                 one per session (section 3)
    warmup.json                  glove sets only (section 4)
    glove/<take>.jsonl           glove frames, full rate (section 5)
    leap/<take>.jsonl            camera frames when the camera ran (section 5)
    events/<take>.events.jsonl   cues and decisions (section 6)
    stills/<take>.png            one hand-cropped IR still per take, camera on
    keypoints/<take>_keypoints.txt   Set A only: professor format (section 7)
    meta/<take>.json             Set A only (section 7)
    rejected/<same layout>       attempts that were not accepted, plus
                                 rejected/<take>.reason.txt
    check.csv, check.txt         written by check_protocol.py (section 8)

Take name: `<item>_<hand>_take<N>_<YYYYMMDD_HHMMSS>`, where `<item>` is the
item id from the protocol file (`index_fast`, `seq4_pairs`, `cylindrical`)
and `<N>` counts accepted takes of that item from 1. A rejected attempt
keeps the number it was trying to become.

## 2. Protocol files (`protocols/`)

All three share: `name`, `version` (integer), `description`, `items` (list).
Every item has `id` (file-name safe, unique) and `label` (words shown to the
operator).

`finger_flexion.json`

    {"name": "finger_flexion", "version": 1,
     "description": "...",
     "takes_per_item": 1,
     "items": [
       {"id": "thumb",      "label": "thumb",  "finger": "thumb",
        "cycles": 5, "bend_s": 4.0, "hold_s": 1.0, "straighten_s": 4.0, "rest_s": 1.0},
       {"id": "index", ...}, {"id": "middle", ...}, {"id": "ring", ...}, {"id": "pinky", ...},
       {"id": "index_slow", "label": "index, slow", "finger": "index",
        "cycles": 5, "bend_s": 6.0, "hold_s": 1.0, "straighten_s": 6.0, "rest_s": 1.0},
       {"id": "index_fast", "label": "index, fast", "finger": "index",
        "cycles": 8, "bend_s": 1.0, "hold_s": 0.5, "straighten_s": 1.0, "rest_s": 0.5}
     ]}

Finger names are always `thumb`, `index`, `middle`, `ring`, `pinky`.

`sequences.json`

    {"name": "sequences", "version": 1,
     "description": "...",
     "hold_s": 2.5, "check_window_s": 1.5,
     "takes_per_item": 3, "shuffle_rounds": true,
     "items": [
       {"id": "seq1_one_at_a_time", "label": "Sequence 1: one finger at a time",
        "steps": [
          {"label": "open hand", "flexed": []},
          {"label": "thumb flexion", "flexed": ["thumb"]},
          ...
          {"label": "all fingers flexed", "flexed": ["thumb","index","middle","ring","pinky"]},
          {"label": "open hand", "flexed": []}
        ]},
       ...
     ]}

The seven sequences, steps verbatim from the professor's message (a step is
the set of fingers flexed at that moment; a finger not listed is straight):

1. `seq1_one_at_a_time`: open; thumb; index; middle; ring; little; all; open.
2. `seq2_reverse`: open; little; ring; middle; index; thumb; all; open.
3. `seq3_alternating`: open; thumb; middle; index; little; ring; open.
4. `seq4_pairs`: open; thumb+index; index+middle; middle+ring; ring+little; all; open.
5. `seq5_progressive`: open; thumb; thumb+index; thumb+index+middle; +ring; +little; full fist; open. ("+little" and "full fist" are the same set, both kept as the professor wrote them.)
6. `seq6_configurations`: open; index only; index+middle; middle+ring; ring+little; all; open.
7. `seq7_flex_release`: open; thumb flex; thumb extend; index flex; index extend; middle flex; middle extend; ring flex; ring extend; little flex; little extend.

"little" in the labels is the `pinky` finger in the data.

`grasps.json`

    {"name": "grasps", "version": 1,
     "status": "placeholder until the reference papers are in hand",
     "description": "...",
     "takes_per_item": 3, "duration_s": 5.0, "prep_s": 5.0,
     "items": [
       {"id": "cylindrical", "label": "cylindrical grasp",
        "source": null, "figure": null,
        "shape": "all fingers wrap a vertical cylinder, thumb opposes",
        "object_implied": true},
       ...
     ]}

`source` and `figure` are filled from the papers; until then they are null
and `status` says so. Labels become the papers' own terms when known.

An item may carry `orientation`: text telling the operator how to turn the
hand for that grasp, shown during MAKE THE GRASP and written to each take's
meta as `orientation_hint` (an item without one gets "palm toward the
camera"). It must be text. A session recorded before the hints were added
can still be resumed when adding them, each as an `"orientation": "...",`
line of its own, is the only change to the file (`session.json` then notes
it under `protocol_changes`).

## 3. `session.json`

    {"set": "finger_flexion",
     "protocol_file": "protocols/finger_flexion.json",
     "protocol_name": "finger_flexion", "protocol_version": 1,
     "protocol_sha256": "...",
     "hand": "left", "operator": "N Kim",
     "camera": "leap" | "none", "glove": true | false,
     "started": "2026-09-28T14:05:01", "ended": "...",
     "xr_trainer_calibrated_at": "2026-09-28T14:02:30" | null,
     "seed": 12345 | null,
     "rounds": [["seq3_alternating", "seq1_one_at_a_time", ...], ...] | null,
     "tool_commit": "960501b" | null,
     "takes": [
       {"item": "index", "take": 1, "name": "index_left_take1_20260928_140530",
        "accepted": true, "reason": "", "decided_at": 1790000000.1,
        "files": {"glove": "glove/....jsonl", "leap": "leap/....jsonl",
                  "events": "events/....events.jsonl", "still": "stills/....png"}}
     ]}

Rejected attempts appear in `takes` with `accepted: false`, a non-empty
`reason`, and paths under `rejected/`.

## 4. `warmup.json` (glove sets)

Open palm 3 s then full fist 3 s, cued by beeps, before the first take.

    {"hand": "left",
     "t_open": [t0, t1], "t_fist": [t0, t1],
     "glove": {"open": {"thumb": 1.43, ...}, "fist": {...}, "span": {...}, "frames": 178},
     "camera": {"open": {...}, "fist": {...}, "span": {...}, "frames": 270} | null,
     "refused": null | "reason"}

Values are medians of the per-finger curl over each window, in the curl
units the rest of the repo uses (`leap_hand.pose_check` and
`cam_hand.fusion`). `span = open - fist` per finger. A glove span under
0.30 on any of index/middle/ring/pinky refuses the session with the reason.

Fraction of a finger's range, used everywhere below:
`fraction = (open - curl) / (open - fist)`, so 0 = the warm-up open palm and
1 = the warm-up fist, clipped to [-0.5, 1.5].

## 5. Frame files

Glove and camera lines are exactly what the coached recorder
(`scripts/record_simultaneous.py --camera leap`) writes today, one JSON
object per line with `wall_time`, `timestamp`, `packet_counter`,
`hand_side`, `frame_id`, `status`, `joints` (26 OpenXR joints with x y z in
metres and a quaternion) and the camera's extra fields, plus three added
top-level keys on every line: `"session"` (the session folder name),
`"item"`, `"take"`. The glove is recorded at its full incoming rate. The
camera file keeps both hands if the tracker reports two; readers filter
on `hand_side`.

## 6. Events file

One JSON object per line, `t` is `time.time()` on the recording machine
(the same clock as `wall_time` in the frame files).

    {"t": ..., "kind": "take_start", "item": "index", "take": 1}
    {"t": ..., "kind": "cue", "step": 0, "label": "bend the index",
     "flexed": ["index"], "hold_s": 4.0, "cycle": 1, "phase": "bend"}
    {"t": ..., "kind": "cue", "step": 1, "label": "hold", "flexed": ["index"],
     "hold_s": 1.0, "cycle": 1, "phase": "hold"}
    {"t": ..., "kind": "cue", "step": 2, "label": "straighten", "flexed": [],
     "hold_s": 4.0, "cycle": 1, "phase": "straighten"}
    {"t": ..., "kind": "cue", "step": 3, "label": "rest", "flexed": [],
     "hold_s": 1.0, "cycle": 1, "phase": "rest"}
    {"t": ..., "kind": "take_end", "item": "index", "take": 1}
    {"t": ..., "kind": "decision", "accepted": true, "reason": "", "by": "auto"}

`step` counts cues from 0 within the take. For sequences there is no
`cycle`/`phase`; `flexed` is the step's set and `hold_s` is the protocol's
`hold_s`. The operator's own redo (`r` during the pause after a take) is a
decision with `by: "operator"`. Every cue is written at the moment the beep
sounds.

## 7. Set A per-take files

`keypoints/<take>_keypoints.txt`: one block in the professor's format
(`scripts/glove/export_prof_format.py` layout: `Frame ... | Hand ID ... |
Time ...`, `Wrist:`, `0 (Palm):` to `20 (Pinky):`, millimetres). The block
is the medoid frame of the take's static interval (`scripts/leap/
record_frame.py` medoid rule applied inside the interval). Coordinates are
the camera's, in millimetres, with the wrist line giving the wrist's
position (not forced to zero, unlike the glove files).

`meta/<take>.json`:

    {"item": "cylindrical", "take": 1, "hand": "left",
     "frames": 452, "tracked_fraction": 0.98, "reacquisitions": 0,
     "gate": {"reacquisitions_in_static_interval": 0, "loss_gap_s": 0.1,
              "losses": [{"t": ..., "duration_s": 1.4, "height_cm": 49,
                          "offset_cm": 8, "view_deg": 20, "cause": "too high",
                          "fix": "keep the palm 25 to 35 cm above the module"}],
              ...},
     "tracker_hand_labels": {"left": 3, "right": 449},
     "static_interval": [t0, t1], "medoid_wall_time": t,
     "grab_strength": 0.91, "pinch_strength": 0.12,
     "curls": {"thumb": ..., "index": ..., ...},
     "orientation_note": "",
     "accepted": true, "reason": "", "decided_at": t}

Static interval: the `static_s` (default 2.0 s) window inside the take with
the smallest mean joint speed, computed on the tracked frames of the
operator's hand. `reacquisitions` counts every hand-id change in the whole
take; `gate.reacquisitions_in_static_interval` counts only the id changes
inside the static interval that came after the hand was really gone (a hole
longer than `loss_gap_s`), which is the number the acquisition gate and the
checker judge on; `gate.id_changes_in_static_interval` counts all id changes
there, including re-labelling with no hole, which does not fail a take.

Which hand. The coached recorder follows the hand OPEN HAND acquired by
the tracker's hand id: a new id within 10 cm of where that hand last was,
in a frame without it, is the same hand (a chirality flip or a
re-acquisition); an id in the same tracking frame, or further away, is
another hand. Without coaching (and in files written before it), the
operator's hand is the label with the most tracked frames. The static
interval and the summary frame are chosen from the operator's frames that
carry the label of the session's `hand` only.

    "operator_hand_ids": [25]          hand ids followed as the operator's
                                       hand, in the order they were taken on
    "other_hand_ids": [24]             other hands in the take's frames:
                                       their lines stay in the leap file,
                                       never measured
    "operator_hand_labels": {"left": 242, "right": 43}
                                       the operator's frames per tracker label
    "operator_hand_label": "left"      the tracker's label on the summary
                                       frame (with no summary frame: the
                                       majority label of the operator's
                                       frames)
    "gate": {"interval_label_frames": 180, "interval_expected_frames": 180, ...}
                                       frames labelled `hand` inside the
                                       static interval, and the frames the
                                       tracker produced over its length
                                       (median framerate times static_s, or
                                       the take when shorter); fewer than half
                                       rejects the take

After the tracked fraction and the losses inside the static interval, the
gate rejects a take with "the tracker fitted the hand as a <label> hand for
the whole take; the <hand> hand cannot be measured from that" when no frame
of the operator's hand carries `hand`, and with "... for most of the static
interval (n of m frames as <hand>) ..." when they are under half. The
checker fails a take whose `operator_hand_label` is not `hand`, and notes
`other_hand_ids`.

`coaching` (null with `--no-coach`) says how the hand got into the grasp:

    "coaching": {"acquire_s": 2.5,              seconds of OPEN HAND, summed
                 "acquire_rounds_s": [2.5],     each OPEN HAND of the attempt
                 "acquired": [{"height_cm": 26.1, "view_angle_deg": 11.9,
                               "offset_cm": 3.2, "hand_label": "left",
                               "hand_id": 25}],  the hand each OPEN HAND took
                 "lost_while_forming": 0,
                 "forming_losses": [{"after_s": 0.6, "height_cm": 25.0,
                                     "view_angle_deg": 73.0, "grab": 0.4,
                                     "hand_label": "left",
                                     "cause": "palm turned away",
                                     "fix": "..."}],
                 "form_s": 4.0,                 MAKE THE GRASP seconds
                 "followed_ids": [25],          ids followed through MAKE
                                                THE GRASP, once it was kept
                 "other_hands": [24]}           ids taken as another hand

`acquire_missing` (what was still wrong) is added when OPEN HAND timed out.
`offset_cm` and `hand_id` in `acquired`, `followed_ids` and `other_hands`
are absent in takes recorded before the recorder followed the hand.

Mock sessions (any recorder run with `--mock`, `--mock-glove` or
`--mock-leap`) carry `"mock": true` in `session.json` and are written under
`recordings/protocol_mock/` by default; the packager refuses them unless
told otherwise.

## 8. Checker outputs

`check.csv` one row per take (accepted and rejected), columns at least:
`set, item, take, accepted, reason, frames_glove, glove_hz, glove_gap_max_ms,
frames_camera, paired_fraction, cued_span_fraction, other_spans (json),
cycles_from_events, cycles_from_peaks, steps_total, steps_pass_glove,
steps_pass_camera, lag_ms_median, verdict`.

`check.txt`: the same as a readable table, then per item the transfer curve,
hysteresis and lag lines (`leap_hand.diagnostics`) where the camera
followed (span >= 0.50), with the counts of frames and cycles that met that
rule.

Verdict rules are in the plan, section 4. Exit code 1 when an accepted take
fails a rule, 0 otherwise.

## 9. Hand-in folder

    xr trainer\grasp and flexion set for professor <YYYY-MM-DD>\
      README.txt
      grasps\<item>_left_takeN.jsonl, <item>_left_takeN_keypoints.txt,
             <item>_left_all_frames.txt, grasps_summary.csv
      finger_flexion\<hand>\<take>.jsonl, camera_<take>.jsonl,
             <take>.events.jsonl, <take>.txt (professor format, all frames),
             flexion_report.txt
      sequences\<hand>\ same layout, sequence_check.csv
      <set>\joint_frames\<take>.csv and joint_frames.pdf   when the session
             was exported with scripts/joint_frames_view.py --session:
             26 rows per take (position in the wrist frame in mm,
             orientation, local axes, bone length, flexion, abduction and
             twist to the parent) and one PDF with the drawing and the
             table per take; the PNG drawings stay in the repo.

No PNG or other image anywhere in the folder.
