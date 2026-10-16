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
    warmup_<HHMMSS>.json         the warm-up of each later run of a resumed
                                 session (sections 3 and 4)
    glove/<take>.jsonl           glove frames, full rate (section 5)
    leap/<take>.jsonl            camera frames when the camera ran (section 5)
    forming/<take>.jsonl         Set A coached takes, since 2026-10-02: every
                                 camera frame from the close cue until the
                                 take starts (sections 5 and 7)
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

The `description` of `finger_flexion.json` and `sequences.json` still says
"gloves on, camera as the reference": it was written when both sets ran the
camera beside the gloves. Since 2026-10-12 they are recorded with the glove
alone (`record_protocol.py --camera none`). The two files are left byte for
byte as they are, because every session stores the file's sha256 and a
session whose protocol file no longer matches is read without its cycle
counts and windows (section 3, `protocol_check.load_protocol`). Which
sensors a session ran is its session.json's `camera` key, never the
protocol's description.

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

Sets B and C (`scripts/record_protocol.py`) also write, since 2026-10-01:

    "warmups": ["warmup.json", "warmup_171012.json"],
                            every warm-up file of the session, in order
    "resumed": ["2026-10-01T17:10:05"],
                            when each later run started (--resume); [] for
                            a session recorded in one run
    "calibrated": ["2026-10-01T16:55:00", "2026-10-01T17:09:40"],
                            each run's XR Trainer calibration time, in order;
                            xr_trainer_calibrated_at keeps the first
    "skipped": [{"item": "thumb", "at": "2026-10-01T17:12:30", "take": 1}],
                            items the operator skipped with `s`, and the take
                            number being recorded when it was pressed
    takes[i].warmup         the warm-up file the take was judged against;
                            an entry without it means warmup.json

and, since 2026-10-13:

    "speed_check": {"since": "2026-10-13", "marks": [0.2, 0.8],
                    "min_bend_share": 0.15, "slow_from_s": 2.0} | null,
                            Set B: the bend-speed rule this session's takes
                            are judged on (section 8, "Bend speed"), with
                            its own numbers. null: a session recorded with
                            --no-speed-check, and every sequences session.
                            A session.json without the key was recorded
                            before the rule and is read as null
    takes[i].check          a Set B take's quick check also holds
                            bend_cue_s and straighten_cue_s (the seconds
                            the cue gave), bend_times_s and
                            straighten_times_s (one per counted bend),
                            speed_rule ("in force", "not in force" or "not
                            a slow cue"), bend_need_s (the least a bend may
                            take; null unless in force) and rejected_for
                            ("span", "bend count", "speed", or null)
    "mock_flags"            has "glove_snaps": true for a rehearsal with
                            --mock-glove-snap, and no such key otherwise

`--resume <folder>|latest` carries a session on in its own folder (`latest`
= the newest folder under `<out-dir>/<set>/` whose name ends with `_<hand>`
and holds a session.json). It is refused when the folder's `set`, `hand`,
`protocol_sha256`, `mock` (true or false, not which mock), `camera` or
`time_scale` differ from the run, when the session.json has no `rounds`,
when the run has `--no-speed-check` and the session a `speed_check` (the
session's own rule is carried on either way, never the run's),
and when its warmup.json has no `units` (a session recorded before the
glove thumb was measured in degrees). `--items`, `--takes` and `--seed` are
refused with `--resume`: the session already fixes them. The saved `rounds`
are reused in order; a planned take is passed over when its item already
has `takes_per_item` accepted takes, and take numbers go on from the
accepted count. Each later run asks for the calibration again, waits for
the glove, acquires, and records a new warm-up as `warmup_<HHMMSS>.json`;
the first run's warmup.json is never rewritten. A skip belongs to the run:
a later run records a skipped item again.

Set A (`scripts/leap/record_poses.py --protocol`) resumes with `--resume
<folder>` and writes `resumed` the same way. It is refused when the
folder's `set`, `hand` or `mock` differ from the run, or when the protocol
file's bytes have changed other than by orientation hints (section 2). The
session's items, durations and `takes_per_item` carry over, except what the
command line gives: `--takes N` sets `takes_per_item` to N from then on, and
every item is recorded until it has N accepted takes, numbering carrying on
from the accepted count. `--resume <folder> --takes 5` therefore tops a
session of three takes per grasp up to five: only takes 4 and 5 of each
grasp are recorded, and session.json then says `"takes_per_item": 5`.

A Set A take entry's `files` also has `forming` (section 7): the path of
its forming file, or null (a `--no-coach` take, or one whose MAKE THE GRASP
wrote no frame). Entries written before 2026-10-02 have no such key.

Set A's session.json, since 2026-10-07, also says which rules of the
coached take the recorder ran (section 7, `recorder_rules`):

    "recorder_rules": "2026-10-07",      the rules of the run that last wrote it
    "recorder_rules_changes": [{"at": "2026-10-07T15:10:02", "from": null,
                                "to": "2026-10-07", "what": "..."}],
                                         when a resumed session changed to them
                                         ([] for a session recorded under them
                                         throughout); the takes recorded before
                                         keep their metas as they were
    "coaching": {..., "lost_wait_s": 4.0, "form_window_s": 8.0,
                 "relock_s": 1.0, "reset_limit": 10,
                 "cues": {"ready": {"beeps": 2, "hz": 520, "ms": 70,
                                    "volume": 0.3, "gap_s": 0.07,
                                    "text": "KEEP HAND OPEN - close on the GO beep"},
                          "go": {"beeps": 1, "hz": 1760, "ms": 450,
                                 "volume": 1.0}}}

The coaching block replaces `form_max_losses` (3, the earlier rules' limit)
with those keys. `lost_wait_s` and `form_window_s` carry over to a resumed
session unless the command line gives `--lost-wait` or `--form-window`.

## 4. `warmup.json` (glove sets)

Open palm 3 s then full fist 3 s, cued by beeps, before the first take.
Since 2026-10-01 (Sets B and C) the fist is followed by a straighten
(1.5 s, low beep) and then each finger on its own, thumb to little finger:
"bend the THUMB only" for 3 s (high beep), then "straighten" for 1.5 s (low
beep).

    {"hand": "left",
     "t_open": [t0, t1], "t_fist": [t0, t1],
     "glove": {"open": {"thumb": 1.43, ...}, "fist": {...}, "span": {...}, "frames": 178,
               "single": {"thumb": 118.2, "index": 0.95, ...},
               "single_span": {"thumb": -96.4, "index": 1.02, ...},
               "t_single": {"thumb": [t0, t1], ...}},
     "camera": {"open": {...}, "fist": {...}, "span": {...}, "frames": 270,
                "single": {...}, "single_span": {...}, "t_single": {...}} | null,
     "units": {"thumb": "degrees, TMC_fe + MCP_fe + IP, larger = more bent",
               "fingers": "curl, wrist to tip over palm length, smaller = more bent"},
     "refused": null | "reason",
     "settle_s": 1.0, "try": 1, "earlier_refusals": []}

Values are medians over each window. `t_open` and `t_fist` are the median
windows (the first `settle_s` of each cued window is left out), and so is
each `t_single` entry: the last `settle_s` of that finger's bend window.
`span = open - fist` and `single_span = open - single`, per finger.

Units. The camera block is the curl for all five fingers, in the units the
rest of the repo uses (`leap_hand.pose_check` and `cam_hand.fusion`). So is
the glove block of a warm-up without `units` (every session before
2026-10-01). With `units`, the glove's index to pinky are still the curl,
and the glove's thumb is its bend in degrees, `T_TMC_fe + T_MCP_fe + T_IP`
from `xr_hand.joint_frames` (`cam_hand.recording_protocol.glove_bends`), so
its `span` and `single_span` are negative. Every glove frame file of a
take is read in the units of the warm-up the take was judged against.

Refusal. A glove span under 0.30 on any of index/middle/ring/pinky refuses
the warm-up with the reason (the first rule). Then a finger whose own bend
is not usable refuses it, named, with what to do: index to pinky need
`single_span` of at least 0.50 of their fist `span`, the thumb at least 35
degrees of bend (`single - open`), e.g. "in the single-finger warm-up the
thumb bent only 12 degrees (needs 35): fold it fully across the palm, tip
to the base of the little finger" (`recording_protocol.single_refusal`).
The recorder does a refused warm-up again, whole, up to 3 times before it
refuses the session; `try` is which try the file holds and
`earlier_refusals` the reasons of the ones before it.

Fraction of a finger's range, used everywhere below:
`fraction = (open - value) / (open - end)`, clipped to [-0.5, 1.5], so 0 =
the warm-up open palm and 1 = the end of the range. For a finger the cue
names (Set B's finger; in Set C the step's flexed fingers) the end is its
own `single` when the warm-up has one and it is usable by the rule above,
else the `fist`; for a finger the cue leaves straight it is the `fist`.
A warm-up without `single` gives (open, fist) for every finger, as before.
The formula works in either direction: the curl falls as a finger bends,
the thumb's degrees rise.

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
decision with `by: "operator"`, and so is `s` during a take, which ends it
at once with the reason "skipped by the operator (s) during the take" (the
attempt moves to `rejected/` like any rejection). Every cue is written at
the moment the beep sounds.

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
in a frame without it, is treated as the same hand (a chirality flip or a
re-acquisition; a rule of id and distance, not a proof of identity); an id
in the same tracking frame, or further away, is another hand. Without coaching (and in files written before it), the
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
                 "ready_pause_s": 1.0,          READY: the open hand held
                                                between the acquisition beep
                                                and the high beep
                 "closed_before_cue": 0,        times the hand closed before
                                                the high beep, each sent back
                 "early_closes": [{"during": "ready", "after_s": 0.3,
                                   "speed_mm_s": 96.0}],
                 "form_s": 4.0,                 MAKE THE GRASP seconds
                 "followed_ids": [25],          ids followed through MAKE
                                                THE GRASP, once it was kept
                 "other_hands": [24]}           ids taken as another hand

`acquire_missing` (what was still wrong) is added when OPEN HAND timed out.
`offset_cm` and `hand_id` in `acquired`, `followed_ids` and `other_hands`
are absent in takes recorded before the recorder followed the hand.
`ready_pause_s`, `closed_before_cue` and `early_closes` are absent in takes
recorded before READY (2026-10-02).

The rules of the coached take, since 2026-10-07. Every meta written since
has `"recorder_rules": "2026-10-07"` at its top level (takes recorded before
have no such key), and its `coaching` has:

                 "forming_lost_count": 1,       losses of the hand while
                                                forming the grasp: each waited
                                                for, and found again in a kept
                                                take (0 when none)
                 "forming_losses": [{"after_s": 0.6, ..., "gone_s": 0.51,
                                     "found_after_s": 1.12}],
                                                as before, plus how long the
                                                hand was gone and when (s from
                                                the GO beep) it was found again;
                                                null for the one not found
                 "ready_losses": [{"after_s": 0.4, "height_cm": 25.0, ...,
                                   "during": "ready"}],
                                                losses during READY, each a reset
                 "resets": 1,                   returns to OPEN HAND: READY
                                                losses and early closes
                 "lost_wait_s": 4.0, "form_window_s": 8.0, "relock_s": 1.0,
                 "reset_limit": 10              the limits it ran with

A loss while forming (no frame of the hand for more than 0.3 s) is waited
for: the hand must be found again (its id, or a new id within 10 cm of
where it was lost, in a frame without it) within `lost_wait_s` of its last
frame and no later than `form_window_s` after the GO beep, and is then held
tracked `relock_s` before the hold, which starts no earlier than `form_s`
after the cue. Not found in time, the attempt is rejected: "lost the hand
while forming the grasp and it was not found again within the 4 s wait for
it: at 25 cm ..." (or "... before the forming window ended, 8 s after the GO
beep: ..."; with earlier losses found again, "lost the hand N times while
forming the grasp and the last time it was not found again ..."). A take
that had a loss is not judged for a close after the cue (the close may have
happened out of sight). An early close is `"during": "ready"` (moving at
`move_speed_mm_s`, section below, during READY: `after_s` from the end of
the READY cue and the frame's `speed_mm_s`) or `"during": "make"` (no frame
that fast in the whole MAKE THE GRASP: `after_s` its length,
`peak_speed_mm_s` and `peak_at_s` the fastest frame after the cue; not
judged when MAKE THE GRASP is under 0.5 s). An early close or a loss during
READY resets the attempt to OPEN HAND, up to `reset_limit` times; the next
one rejects it: "closed before the GO beep N times; reset to OPEN HAND 10
times, the limit that ends a stuck loop; the last ..." (with "and lost the
hand during READY M times" after the count when there were such losses).
The static hold and its gate (`gate`) are the same under both rules.

`lost_while_forming` is not written under these rules: it counted the
send-backs of the earlier rules, each followed by "close it slower".

Before 2026-10-07 a loss during READY was in `forming_losses` with
`"during": "ready"` (its `after_s` from the acquisition beep), every loss
sent the operator back to OPEN HAND (`lost_while_forming` counting them),
and losses and early closes together
counted towards three send-backs that rejected an attempt, with the reason
"lost the hand N times while forming the grasp, ...", "closed before the
cue N times, the last ...", or the first followed by "; " and the second.

The forming file, since 2026-10-02. `forming/<take>.jsonl` holds every
frame the camera reported from the close cue (the high beep that starts
MAKE THE GRASP, `coaching.form_start`) until the take's own file opens:
the lines of section 5, exactly as in `leap/<take>.jsonl`, both hands if
two are in view, with two differences: a hand tracked for less than 0.3 s
is kept (the take drops it as settling), and a frame the camera saw before
the cue is not. A reset (a hand that had closed before the cue; before
2026-10-07 also a loss while forming) sends the operator back to OPEN HAND
and the next round starts the file again, so it holds the MAKE THE GRASP
that led to the take; an attempt that is not kept takes its file to
`rejected/forming/`. Since 2026-10-07 a hand lost while forming is waited
for, and the file runs on through the loss, the gap and the hand's new id
in it: such a file then carries `"forming_lost_count": N` (the number of
losses, `leap_hand.transition.FORMING_LOST_KEY`) on every one of its lines,
written when the attempt ends, so the file says so wherever it is read.
Every reading of a file so marked gives no time: `transition_s` null with
the note below, and the hand-in folder's row invalid with that note and no
start, settle or duration (`leap_hand.transition.transition_timing`). The
meta's `files.forming` is its path (null with
`--no-coach`, or when no frame was written), and `coaching` gains:

                 "form_start": 1790961743.818346,  the close cue, wall clock
                 "forming_frames": 152,         lines in the forming file
                                                (0 when there is none)
                 "forming_tracked_fraction": 0.99,
                                                the followed hand's tracked
                                                lines over its median
                                                framerate times cue-to-take,
                                                as tracked_fraction is for
                                                the take; null without a file
                 "transition_s": 0.84,          close cue to settled after
                                                the close; null with a note
                 "peak_speed_mm_s": 187.6,      the fastest frame after the
                 "peak_at_s": 0.102,            cue, and when (s from it)
                 "last_move_s": 0.225,          the last frame at
                                                move_speed_mm_s or faster;
                                                null when none was
                 "transition_note": "",         why transition_s is null, or
                                                what came after it
                 "transition_rule": {"speed_mm_s": 20.0, "window_s": 0.3,
                                     "step_s": 0.1, "move_speed_mm_s": 50.0}

`transition_s` (`leap_hand.transition.settle`) is measured on the forming
file's frames of the hand OPEN HAND acquired, followed by id as the take
is, labelled as the session's `hand`, on the `capture_time` clock. Each
frame's speed is the mean displacement of the 26 joints (`abs26`) across
the `step_s` centred on it, over the time between those frames. The close
is the hand's fastest frame after the cue (`peak_speed_mm_s` at
`peak_at_s`). The hand has settled at the first frame after that peak from
which the speed stays below `speed_mm_s` for `window_s`, with no hole
longer than 0.1 s; searched for after the peak, so a twitch on the beep
and a pause before the close are not the settle. `transition_note`:

    ""                                     a close, settled, nothing after
    "no close after the cue (peak X mm/s): the hand was already in the grasp"
                                           peak_speed_mm_s under
                                           move_speed_mm_s: closed before
                                           the cue; transition_s null
    "settled, then moved again at T s"     a frame at move_speed_mm_s or
                                           faster after the settle (from T s,
                                           until last_move_s); transition_s
                                           is the settle after the close
    "never settled before the hold"        no settle after the close before
                                           the take began; transition_s null
    "no frame of the hand after the cue to measure"
                                           every number null
    "the hand was lost while forming the grasp (N times): a change of pose is not timed across a gap in the tracking"
                                           the file is marked forming_lost_count
                                           (above); every number null

20 mm/s is about three times a held grasp's own speed (6.4 mm/s median
over the static intervals of the 63 takes of session
20261001_101621_left). 50 mm/s is above the fastest frame after the cue of
the two takes of that session closed before it (38 and 43 mm/s) and under
a third of the slowest real close's peak (180 mm/s); see the module
docstring of `leap_hand.transition`. Takes without a forming file (every
take before 2026-10-02, and `--no-coach`) have none of these keys. Takes
whose meta has `transition_s` but not `peak_speed_mm_s`, `last_move_s` and
`transition_note` were timed by the first rule (settled after the first
frame at 20 mm/s); `scripts/leap/recompute_transition.py <session>`
rewrites those six keys of every meta with a forming file (kept and
rejected) from the forming file and `form_start`, and nothing else in the
session.

Mock sessions (any recorder run with `--mock`, `--mock-glove` or
`--mock-leap`) carry `"mock": true` in `session.json` and are written under
`recordings/protocol_mock/` by default; the packager refuses them unless
told otherwise.

## 8. Checker outputs

`check.csv` one row per take (accepted and rejected), columns at least:
`set, item, take, accepted, reason, warmup, frames_glove, glove_hz,
glove_gap_max_ms, frames_camera, paired_fraction, cued_span_fraction,
other_spans (json), cycles_from_events, cycles_from_peaks, steps_total,
steps_pass_glove, steps_pass_camera, lag_ms_median, verdict`. `warmup` is
the warm-up file the take is checked against (Sets B and C; section 3):
each take's fractions use its own warm-up's endpoints (section 4), and its
glove file is read in that warm-up's units. `cued_span_fraction` is on the
cued finger's range; `other_spans` stay on the fist range.

`check.txt`: the header lists every warm-up file of the session with its
glove and camera open, fist and (when it has them) single values, then the
same as `check.csv` as a readable table, then per item the transfer curve,
hysteresis and lag lines (`leap_hand.diagnostics`) where the camera
followed (span >= 0.50), with the counts of frames and cycles that met that
rule. A glove thumb in degrees is negated for the curve and the lag, so it
falls as the thumb bends, as the camera's curl does, and the report says so.

Verdict rules are in the plan, section 4. Exit code 1 when an accepted take
fails a rule, 0 otherwise.

Bend speed (Set B, since 2026-10-13). `recording_protocol.bend_moves` times
every bend the cycle count finds (the same trigger, so as many as
`cycles_from_peaks`). A bend's rise runs from the reading's lowest value
since the finger was last open to its highest before it opens again; the
bend time is from where the reading last left 20 % of that rise to where
it first reached 80 % of it, each crossing placed between the two frames
around it, so a ramp of T seconds reads 0.6 T and a jump inside one frame
0.6 of the frame. The straighten time is the same on the way down, null
for a take that ended before the finger opened. `check.csv` holds
`bend_cue_s` and `straighten_cue_s` (the seconds the cue gave),
`bend_times_s` and `straighten_times_s` (json, one per counted bend),
their medians `bend_time_median_s` and `straighten_time_median_s`, and
`speed_rule` (`in force`, `not in force`, `not a slow cue`). `check.txt`
has the two medians in its table (`bend s`, `open s`), every time in the
per-item lines, and the session's rule in its header (`speed`).

The rule: a bend whose cue the protocol gives `slow_from_s` (2) seconds or
more is a slow bend and must take at least `min_bend_share` (0.15) of the
cue's time, 0.60 s of a 4 s bend and 0.90 s of a 6 s one; the fast index
(1 s) is not judged. In a run at `--time-scale` s a cue is slow by what
the protocol wrote (its seconds over s) and the bar is the share of the cue
as it ran. The recorder's quick check rejects a take with any bend under
the bar, after the span and the bend count (`rejected_for: "speed"`; the
reason reads "the index bent too fast on 5 of 5 bends: 0.05 to 0.13 s, and
a 4 s bend cue needs at least 0.60 s (timed from 20 to 80 percent of each
bend's rise)"). The checker fails an accepted take the same way ("index
bent too fast: 5 of 5 bend(s) took ...") only in a session whose
session.json has `speed_check`, and by that session's own numbers. In a
session without it the verdict is what it was, the times are reported, and
the take's notes say which bends the rule of 2026-10-13 would have refused.

The straighten is timed and never judged. Where the bar comes from, as
`bend_moves` reads the recordings on disk: in 20261013_180909_left,
recorded an hour before the rule, the 25 bends under 4 s cues took 0.05 to
0.43 s and the 5 under the 6 s cue 0.42 to 0.55 s, at most 0.11 of their
cue. In the one recording of a finger bent slowly on purpose
(results/diagnostics/dead_zone_right_ring_20260920_212254.csv,
`finger_sweep.py`, 5 s to bend and 5 s to straighten, right ring) the three
paced bends read 1.19, 1.74 and 3.32 s, 0.24 of the cue or more; a fourth
rise, 0.11 s while the finger was opening, is a stray that the bend count
already fails a take for. That sweep's straightens read 0.09, 1.05 and
0.96 s of 5 s, and the straightens of 20261013_180909_left 0.05 to 1.91 s:
the glove's reading falls quickly once a finger starts to open, so a bar on
the straighten would not tell a slow opening from a fast one. The bend's
bar rests on that one sweep of one finger of one glove, which is why
`--no-speed-check` exists.

Set A, since 2026-10-02: a take with a forming file (section 7) fails
unless its meta's `coaching` has the key `transition_s` (null allowed) and
`forming_frames` of at least 1; its `notes` then carry "transition 0.84 s,
close cue to settled (152 forming frames)" (followed by "; " and the
`transition_note` when it has one), or "transition none: <the
transition_note> (152 forming frames)" for a null, in check.csv and
check.txt. A null with its note is a pass. A meta without
`peak_speed_mm_s`, `last_move_s` or `transition_note` passes with the note
"recompute needed: ..." naming the missing keys and
`scripts/leap/recompute_transition.py`, and the header counts those takes.
A take without a forming file is checked as before.

## 9. Hand-in folder

    xr trainer\grasp and flexion set for professor <YYYY-MM-DD>\
      README.txt
      validation_report.txt   (a grasp-only folder built by
             scripts/rebuild_grasp_set.py)
      grasps\<item>_left_takeN.jsonl, <item>_left_takeN_keypoints.txt,
             <item>_left_all_frames.txt, grasps_summary.csv,
             grasp_types_audit.csv, joint_angles_24.csv
      grasps\joint_frames\<item>_left_takeN.csv, joint_frames.pdf,
             joint_key.csv, joint_key.pdf (the joint key,
             `xr_hand.joint_key`, docs/joint_frames.md)
      grasps\images\<item>_left_takeN.png (one per accepted take whose
             still has the recorded hand's skeleton drawn on it, cut to that
             hand by `leap_hand.grasp_images`), grasp_images.pdf (one page
             per grasp type with a picture)
      transitions\pose_transitions.csv,
             open_hand_to_<item>_left_takeN.jsonl (a take's forming file,
             when it has one), <from>_to_<to>_<hand>_takeN.jsonl (the takes
             of a transition session, section 10)
      finger_flexion\<hand>\<take>.jsonl, camera_<take>.jsonl (only for
             a session that ran the camera), <take>.events.jsonl, <take>.txt
             (professor format, all frames), joint_angles\<take>.csv (every
             glove frame: its cue and the 24 angles), cue_angles.csv (the 24
             angles at every held cue), flexion_report.txt,
             <session>_warmup.json and <session>_warmup_<HHMMSS>.json (every
             warm-up of the session)
      sequences\<hand>\ same layout, sequence_check.csv in place of
             flexion_report.txt
      <set>\joint_frames\<take>.csv and joint_frames.pdf   when the session
             was exported with scripts/joint_frames_view.py --session:
             26 rows per take (position in the wrist frame in mm,
             orientation, local axes, bone length, flexion, abduction and
             twist to the parent) and one PDF with the drawing and the
             table per take; the PNG drawings stay in the repo.

`scripts/package_professor_set.py` writes it; for the grasp set alone
`scripts/rebuild_grasp_set.py` is the one command (joint frames, the
folder, the check). What the folder says about Set A is worked out by
`leap_hand.grasp_handin` from one pass over the accepted takes, each
measured from its own lines by `leap_hand.grasp_audit`. File names in the
tables are relative to the hand-in folder, with forward slashes.

`grasps_summary.csv`, one row per handed-in take:

    grasp_name, grasp_label, source_paper, source_figure, hand, take
    raw_jsonl_file, representative_keypoints_file, all_frames_file,
      joint_frames_file, image_file, transition_capture_file
                            the take's files; image_file is its picture
                            under grasps/images/, empty when its still does
                            not show the hand
    sensor                  "camera" (leap file, null glove) or
                            "glove+camera"
    recording_duration_s    last minus first wall_time of the labelled
                            hand's lines. NEVER from a count of lines: a
                            take with two hands in view has two lines per
                            frame
    camera_clock_duration_s the same span on LeapC's timestamp_us
    recorder_window_s       take_end - take_start of the meta
    unique_frame_count      distinct frame_id in the file
    hand_frame_count        lines of the labelled hand
    jsonl_row_count         lines, every hand
    frame_rate_hz, tracked_percent, largest_gap_s
    static_interval_start_s, static_interval_end_s,
      static_interval_duration_s      from the labelled hand's first line
    static_interval_pose_speed_mm_s    the median pose speed inside it
                            (`leap_hand.transition.pose_speeds`: the
                            joints' speed with the whole hand's movement
                            taken out), about 1 mm/s for a held pose
    static_interval_first_frame_id, static_interval_last_frame_id
                            the tracker frame numbers the static interval
                            runs between: the 'Frame' numbers of the
                            all-frames text file, so a reader can cut it
    take_shape_rms_max_mm, frames_within_5mm_percent, shape_rms_start_mm,
      shape_rms_end_mm      the whole recording against its summary frame
                            (RMS over the 26 joints after a rigid fit): the
                            furthest frame, the share of frames within 5
                            mm, and the mean over the first and the last
                            0.2 s. A recording is five seconds and not all
                            of it is the pose; an end more than 10 mm away
                            gets a note (`leap_hand.grasp_audit.
                            whole_take_shape`)
    static_interval_mean_speed_mm_s, static_interval_shape_rms_mm,
      static_interval_shape_rms_max_mm   how still the static interval
                            was: the mean joint speed over it in the camera
                            frame, and the RMS distance of the 26 joints
                            from the summary frame after a rigid fit (the
                            median over the window, and the largest).
                            Measured, never a gate
                            (`leap_hand.grasp_audit.stillness`)
    representative_frame_index      the summary frame's line in the JSONL,
                            from 0 (the meta's medoid_line)
    representative_frame_id, representative_frame_time_s
    other_hand_detected, other_hand_frame_fraction,
      other_hand_median_distance_cm   recounted from the lines, not read
                            from the meta's other_hand_ids
    other_hand_finger_curl_minus_own_deg   the sum of the MCP, PIP and DIP
                            flex_deg of the index, middle, ring and little
                            finger, the other hand's minus the labelled
                            hand's, each at its frame nearest the middle of
                            the recording; empty when the other hand is in
                            view on fewer than 50 % of the frames
                            (`leap_hand.grasp_audit.other_hand_curl`)
    hand_selection          "followed id" (operator_hand_ids in the meta)
                            or "tracker label"
    hand_ids                the tracker ids of the labelled hand's lines
    tracker_label_on_representative_frame, tracker_id_changes
    identity_flag           "" or why the hand's identity needs a note: an
                            id change, two labels, the summary frame under
                            the other label, a second model fitted on the
                            same hand (another "hand" within 10 cm)
    status ("accepted"), decided_by, reacquisitions_in_static_interval,
      reacquisitions, grab_strength, pinch_strength, curl_*,
      palm_height_cm, view_angle_deg      copied from the take's meta
    index_metacarpal_mm, index_metacarpal_range_mm   the length LeapC gave
                            the index metacarpal (the median over the take)
                            and how much it changed inside the take. The
                            tracker refits the hand's size, so millimetres
                            are not comparable between takes as they stand
    first_wall_time, static_interval_start, static_interval_end,
      medoid_wall_time      the same moments as epoch seconds
    orientation_note, session, recorder_take_name, notes
                            notes = the take's timing notes, its pose notes
                            (a static interval faster than 20 mm/s, a shape
                            more than 5 mm RMS from the summary frame, a
                            fit with the back of the hand to the camera),
                            its identity flag, what the still notes
                            (below) read in its picture
                            (`grasp_handin.picture_notes`) and, for a take
                            kept after the hand was lost while forming
                            (rules of 2026-10-07), "the hand was lost N
                            time(s) while forming the grasp and found again
                            before the hold; the forming file is not timed"
                            (`grasp_handin.forming_lost_notes`); a note is
                            never a rejection

The README lists every one of these columns with what it holds
(`grasp_handin.COLUMN_HELP`; a column without an entry fails the build),
and says which are copied from the meta and which are measured again.

No column of it carries a transition time. `grasp_types_audit.csv` has one
row per grasp of the protocol, recorded or not: its paper and figure,
`same_configuration_as` (the protocol's `same_as` links followed both ways,
so every member of a configuration lists the others), the accepted
recordings and takes against `target_takes`, how many of each dataset
exist, `valid_transitions` and `questionable_transitions`, how many of its
recordings carry no note (`recordings_without_a_note`; 0 with recordings
means every one of them has something a reader must know first), how many
have a picture (`recordings_with_an_image`), how far
its takes are from each other (`between_take_max_spread_deg`, the largest
spread over its takes of one of the 18 given angles, and
`between_take_max_spread_angle`, which angle), the attempts not kept
(`attempts_not_kept` = `open_hand_timeouts` + `rejected_recordings`: a wait
for an open hand that ran out is not a rejected recording, and
`most_common_reject_reason` leaves the waits out), and two statuses:
`static_pose_status` READY (one take with keypoints, the frame-by-frame
file, the joint-frame table and the angles) or INCOMPLETE, and
`multi_take_status` COMPLETE or INCOMPLETE against the target. They are two
questions and are never merged.

`joint_angles_24.csv` has one row per take with a joint-frame table, the
paper's 24 names in the paper's order. 18 are filled. The six a camera take
does not give (`xr_hand.joint_frames.CAMERA_NOT_GIVEN`: the four finger
CMC, which LeapC's rigid palm does not measure, and the thumb's TMC pair,
which is about other axes than the paper's) are EMPTY, never a number
under a name it does not deserve. What the tracker does measure at those
joints follows under honest names (`grasp_handin.METACARPAL_COLUMNS`):
`T_metacarpal_flex_wrist_ref`, `T_metacarpal_abd_wrist_ref` and
`{I,M,R,L}_metacarpal_flex_wrist_ref` (the METACARPAL rows' own cells:
where each metacarpal points seen from the wrist joint, which includes the
wrist's bend), the same six with `_palm_ref` (seen from the palm joint:
constants for the four fingers), `forearm_to_palm_deg`, and the four MCP
pairs in the paper's own rotation order (`*_paper_order`,
`xr_hand.joint_frames.paper_order_angles`). The PDF prints "n/a" for the
six on every camera page.

`transitions\pose_transitions.csv` has one row per recorded change of pose
and none for a take without one: participant_id, hand, from_grasp
("open_hand" for a coached take's forming file), to_grasp, take,
recording_file, cue_wall_time, cue_time_s, movement_start_time_s,
settled_time_s, transition_duration_s (= settled - movement start, only on
a row whose validity is valid or questionable), reaction_time_s,
cue_to_settled_s, peak_speed_mm_s, speed_at_cue_mm_s, pose_change_mm,
hand_speed_at_cue_mm_s, hand_peak_speed_mm_s, last_movement_time_s,
movement_start_method, settled_method (the rule in words with the
thresholds it was run with), validity, notes, frames, hand_ids,
first_frame_capture_time, analysed_until_s, session, static_take_file. The
times are `leap_hand.transition.transition_timing`'s, in seconds from the
first measured frame of the recording. A static take is never timed. A
forming file marked `forming_lost_count` (section 7) has a row that is
invalid, with the notes "the hand was lost while forming the grasp (N
time(s)): ..." and no start, settle, duration or speed; README section 17
counts such rows among the invalid, section 15 lists their takes under a
count line, and section 5 says that the rules changed, for how many takes,
read from the takes' metas (`recorder_rules`, `coaching.forming_lost_count`)
and only when some take was recorded under the new rules.

What is timed is the POSE, not the hand over the camera. Every speed in
the table is the pose speed (`transition.pose_speeds`: of the two frames a
speed is taken between, the later is laid on the earlier by the rigid fit
of the palm's 11 joints, and the mean displacement of the 26 joints that
is left is divided by the time), except the two `hand_*` columns, which
are the speed over the camera and decide nothing. The movement is the
stretch of pose speed at 25 mm/s or faster, 0.1 s or longer, that holds
the fastest frame after the cue; settled is the first frame after its
fastest from which the pose speed stays under 10 mm/s for 0.3 s; valid
needs a pose that was still at the cue, one such movement and a settle;
questionable is the same with a stir of the pose before the main movement
or a grip adjusted after the settle. `pose_change_mm` is the RMS distance
of the 26 joints between the pose at the cue and the pose at the settle
(palm fit): the size of the change, which the reading depends on. So
`cue_to_settled_s` is no longer the meta's `transition_s`: that one is the
recorder's own, on the speed over the camera. The README shows every timed
recording at twice and at half the two speed thresholds as well, because
the length depends on them by a factor of three.

The README of a folder with Set A says under its title "Images: one
infrared picture per recording, cut to the recorded hand, grasps\images
(section 5). No other photographs are included." (with "whose still shows
the hand (n of N)" after "per recording" when not every take has a
picture, and "No photographs are included." when none has), opens with
where each thing that was asked for is
(`leap_hand.grasp_handin.ASKED_FOR`: ten items, the last two the images of
the grasp types and which joint each angle corresponds to, each with its
file and its section) and then has twenty numbered sections
(`leap_hand.grasp_validate.SECTION_TITLES`), every count in them computed
at the build, followed by the glove sets' part when Sets B or C are in the
folder. Section 1 says what was seen in the hands, read by eye on the recorder's
UNCUT stills (both hands and the room; kept in the session folder, never
handed in), from the still notes (below), and says that the folder's own
pictures are cut to the recorded hand, so what is reported for the other
hand cannot be seen in them; section 5, THE PICTURES, gives the cut rule
in six steps with `leap_hand.grasp_images`' numbers; section 11 prints the
joint key (`xr_hand.joint_key.readme_lines`); section 15 lists the picture
notes under three kinds (something in or against the recorded hand,
something in the other hand in the uncut still and not in the cut picture,
the still does not settle it), and a fourth, the picture has not been read
by eye, when a picture is not in the notes' `read`; the captions of
`grasp_images.pdf` carry the same readings, marked "uncut still", and an
other-hand reading is marked "not in this picture";
section 16 gives the
measured other-hand finger curl (`other_hand_finger_curl_minus_own_deg`).
A rebuild leaves a file whose bytes would not change untouched,
and moves one it changes, or no longer makes inside a set's folder, to
`_superseded\<folder name>\<time>\` beside the hand-in folder; it deletes
nothing and only reads the sessions. `scripts/validate_grasp_set.py`
checks a built folder against its own files and reports STATIC-POSE TASK,
FULL MULTI-TAKE DATA COLLECTION (our replication target, and labelled so),
TRANSITION TIMING (PARTIAL with a valid timing of a recorded change,
MISSING without one, never more) and FOLDER CHECK; its exit code follows
FOLDER CHECK only. Every joint-frame table, every column of the angle
table, the stillness and the pose speed of every static interval, the
shape of every whole recording and the hand's size it computes a second
time with its own implementation of the README's formulas (`readme_table`,
`readme_angle_row`, `readme_stillness`, `readme_pose_speed`,
`readme_whole_take`, `readme_hand_size` in `leap_hand.grasp_validate`),
which shares no code with the library that made them; the choice of the
static interval, of the summary frame and the transition rule it runs
again with the library's own functions. It checks the pictures (every
`image_file` there, a PNG that opens and is not all black,
`recordings_with_an_image` per grasp, `grasp_images.pdf` with one page per
grasp type with a picture, no other image in the folder), the joint key
when there are joint-frame tables (`joint_key.csv` with the 26 rows in
table order and every given angle of the paper in its row and column,
`joint_key.pdf` one page, the naming rule and every row in README section
11), the README's line on images, that section 1's counts of what the
pictures show add up to the summary's rows (a row noted "has not been read
by eye" counted on the not-yet-read line), and that section 1 does not say
"No object was held" nor section 16 "held open"; its report counts "pictures
cut to the hand: PNGs that open and are not black", "recordings without a
picture", "grasp_images.pdf pages" and "joint key rows".

Section 3 of the README prints the lines of
`protocols/<protocol>_source_notes.txt` when that file exists: what we read
into the papers beyond their own words (which photograph of a figure is
which sub-class, the order of a strip of photographs). It sits beside the
protocol file and not in it, because a session pins its protocol by hash.

`protocols/grasps_still_notes.json` is what the stills of one session show
in the hands, read by eye: `what` (the file in words), `session` (the
session it is of, e.g. `20261001_101621_left`), `read_on` (the day),
`confirmed_by_operator` (true only when the operator has confirmed the
readings; false in the file now), `in_or_against_the_hand`
and `in_the_other_hand` ({stem: text}), `cannot_tell` ([stem]), where a
stem is `<grasp>_<hand>_take<N>`, and `read` ([stem], session order): the
takes whose picture was looked at; a picture not in it has not been read, and
section 1 lists it as not yet read by eye, never as "nothing seen" (without
`read`, the takes named are the ones read). An optional `other_hand_by_eye`
(text) is printed in section 16. `leap_hand.grasp_images.load_still_notes`
reads it; the build uses it only for the session it names (`grasp_handin.load_notes`)
and otherwise says the pictures were not read. Its readings are notes,
never measurements.

Images: the only ones the folder may hold are the planned
`grasps\images\<item>_<hand>_takeN.png`, written by this build from a still
(`package_professor_set.allowed_images`). Every planned file is checked
before anything is written, by its name and, for a copy, by its source's
first bytes (`image_violations`); the folder is checked before and after
writing (`stray_images`, by name or first bytes, however deep). Any other
image and the packager prints `REFUSED`, names it and exits 1. The
joint-frame PNGs and the stills never go in as they are; Sets B and C get
no pictures.

### Sets B and C: glove only or glove and camera, and the angle tables

A glove set's README, its flexion report and its known limits are written
from each session's `camera` key (`protocol_check.ran_camera`). A folder
whose glove sessions all have `"camera": "none"` says "gloves on, no
camera", lists no `camera_<take>.jsonl`, and makes no statement about a
camera reference, a transfer curve, hysteresis or lag; a folder that mixes
the two kinds names the glove-only sessions; every session's lines carry
`sensors: glove only, no camera` or `sensors: glove and Ultraleap camera`.

`<set>\<hand>\joint_angles\<take>.csv` (`cam_hand.angle_series`), one row
per glove line of the recorded hand in the take's glove file, in time order:

    frame_id      the line's frame_id; XR Trainer's glove lines carry 0, so
                  it does not number the frames (corrected 2026-10-16; it
                  was described as the glove's own frame counter)
    wall_time     the line's wall_time, seconds since 1970, 6 decimals: when
                  the recorder wrote the line. The .jsonl line also has
                  capture_time, when the glove's packet reached the
                  computer (QueueItem.recv_time), which can be earlier
    t_s           wall_time minus the take_start event's t (the first cue's,
                  or the first frame's, when there is none), 4 decimals
    cue_step, cue_label, cue_phase, cue_cycle, flexed
                  the cue in force: the last cue event whose t is at or
                  before wall_time. Empty before the first cue and from
                  take_end on. cue_phase and cue_cycle are Set B's (bend,
                  hold, straighten, rest; 1 to the item's cycles); flexed is
                  the cue's fingers separated by spaces, empty for none
    T_TMC_fe T_TMC_aa T_MCP_fe T_IP, then for I, M, R, L:
    <f>_CMC <f>_MCP_fe <f>_MCP_aa <f>_PIP <f>_DIP
                  the 24 named angles of `xr_hand.joint_frames.dof24` on the
                  line's joint table, degrees, 2 decimals (the same table and
                  signs as the grasp set's joint_angles_24.csv)

`<set>\<hand>\cue_angles.csv`, one row per held cue of every handed-in take
of that hand, session after session:

    session, take_name, item, take
    step, label, phase, cycle, flexed      the cue (as above)
    cue_t_s                 the cue's t minus the take's time zero
    window_start_s, window_end_s
                            the window the row is read over. Set B: a hold
                            or a rest, its whole phase (cue t to t + hold_s;
                            the bend and the straighten have no row). Set C:
                            every step, the last check_window_s / hold_s
                            share of its hold, `protocol_check.step_windows`'s
                            window, so the row and the step's row in
                            sequence_check.csv are read over the same frames
    frames                  glove frames with window start <= wall_time <
                            window end
    the 24 angles           the median of each over those frames, 2 decimals;
                            empty when the window holds no frame

`flexion_report.txt` gives each handed-in Set B take two more columns,
`bend s` and `open s`: the take's median bend time and median straighten
time over the seconds the cue gave (`0.17/4`), with one line under each
session saying whether its bends were judged on the rule (section 8, "Bend
speed"). The README's QUALITY RULES state the rule with the sessions' own
numbers and name the sessions recorded under it and those that were not; a
session's line in SESSIONS says the same.

Nine of the 24 are not measurements (`angle_series.MODEL_ONLY`): the four
finger CMC and the five abductions (T_TMC_aa and each finger's MCP_aa) are
values of the glove's hand model, written as streamed. The README names
them and prints the lowest and highest value of every one of the 24 over
the handed-in frames of each set and hand, marking the nine.

The README also prints, per set, hand and digit, whether one value gives all
of the digit's flexions (`angle_series.one_value`): the frames are grouped by
the digit's summed flexion (thumb T_TMC_fe + T_MCP_fe + T_IP, a finger
MCP_fe + PIP + DIP) in steps of 1 degree; inside each group of 3 frames or
more every joint's flexion is compared with the group's mean; the line gives
the span of the sum, the RMS of those differences for the joint where it is
largest, that joint and the frames counted. A digit whose sum moved less
than 10 degrees, or with under half of its frames in a group that counts,
is "not judged". A fraction of a degree means the digit's three flexions
are one measurement spread over three joints by the glove's hand model (on
the real glove take of 2026-10-01: 0.05 to 0.11 degrees while the sums moved
51 to 157 degrees), so a dependency between joints of one digit cannot be
learned from the files, while dependencies between digits are measured.

## 10. Transition sessions

`scripts/leap/record_transitions.py` records a cued change from one held
pose into another, camera only, into a session of its own:

    recordings/protocol/transitions/<YYYYMMDD_HHMMSS>_<hand>/
      session.json           "set": "transitions", hand, operator,
                             participant, mock, pairs, takes_per_pair, the
                             holds and waits, "thresholds", and "takes":
                             name, from_grasp, to_grasp, take, attempt,
                             accepted, reason, files {leap, meta}
      pose_transitions.csv   one row per attempt, section 9's columns
                             (recording_file relative to the session)
      leap/<from>_to_<to>_<hand>_take<N>_<stamp>.jsonl
                             every frame of every hand from `pre_s` before
                             the cue to the end of the hold, section 5's
                             camera line format, pose "<from>_to_<to>"
      meta/<same name>.json  cue_wall_time, start_hand_id, other_hand_ids,
                             start_palm (the hand that was held still at
                             the cue), cue_wait_s, early_moves, frames,
                             thresholds, "timing" (every field of
                             `leap_hand.transition.Timing`), accepted,
                             reason

A take is accepted when its timing is valid or questionable. Frames from
before the cue are written at the cue, so their `wall_time` is that write;
`capture_time` is the clock of every timing. A mock session goes to
`recordings/protocol_mock/transitions/` and is left out of a hand-in unless
`--allow-mock`; when it is let in, the README's MOCK DATA banner names it,
every one of its rows in `pose_transitions.csv` has notes starting
`MOCK (synthetic hand): `, section 17 marks its takes, and the validator
refuses the folder unless it too is given `--allow-mock`.
`scripts/rebuild_grasp_set.py` copies the accepted takes of every real
session under `recordings/protocol/transitions/` (its `--transitions-dir`
default, so the Desktop launcher needs no argument) and of any folder given
with `--transitions <session>` into `transitions\`, and times them again
with the build's thresholds.
