# Recording the grasp set (Set A)

The runbook for Set A of the professor's protocol
(`docs/grasp_and_flexion_protocol_plan.md`): every grasp in
`protocols/grasps.json` held still over the Ultraleap camera with the bare
left hand, 3 takes of 5 seconds each, no object in the hand. The recorder is
`scripts/leap/record_poses.py --protocol`. What it writes is fixed by
`docs/protocol_formats.md` (sections 1, 3, 5 and 7):

    recordings\protocol\grasps\<YYYYMMDD_HHMMSS>_left\
      session.json                     the whole session, rewritten after every take
      leap\<take>.jsonl                every camera frame of the take
      forming\<take>.jsonl             every camera frame from the GO beep of MAKE
                                       THE GRASP until the take starts (since 2026-10-02)
      stills\<take>.png                one IR picture of the hand, cropped to the hand
      keypoints\<take>_keypoints.txt   the take's summary frame, professor's format
      meta\<take>.json                 the numbers and your decision
      rejected\                        takes not kept, same layout, plus <take>.reason.txt

A take is named `<grasp>_left_take<N>_<YYYYMMDD_HHMMSS>`, for example
`hook_left_take2_20260928_141503`.

## What happens in one take

1. The console names the grasp, describes its shape, says which way to turn
   the hand for it (the orientation rule below) and repeats the orientation
   envelope.
2. OPEN HAND: hold your open hand still over the camera, palm toward it,
   until you hear two short, quiet, low beeps.
3. READY: those two short low beeps are not the cue. Keep the hand open and
   still for 1 more second while the window says "KEEP HAND OPEN" and
   "close on the GO beep".
4. MAKE THE GRASP: on the GO beep, one long, loud, high beep, form the grasp
   in the picture naturally and smoothly while keeping the palm visible to
   the cameras (a deliberately slow close would bias the transition timing).
   Close only on the GO beep.
5. HOLD STILL: at the next beep, hold the grasp still until the low beep.
6. The recorder checks the take against the acquisition gate (below). A take
   that fails is moved to `rejected\` and recorded again on its own.
7. A take that passes shows the review in the COPY THIS window: the still
   of your hand and the numbers. You decide with one key.

The camera can follow an open hand into a grasp. In this setup the tracker
often failed to reacquire a hand that had already closed and was occluded
(closed hands are tracked; the losses are self-occlusion during the close).
That is why the hand opens first, every time.

OPEN HAND waits until the palm has been 18 to 40 cm above the module, within
50 degrees of facing the lens and over the middle of the module for half a
second without a break. The COPY THIS window says what is still wrong
("lower it", "centre it over the module", "turn the palm to the camera"). If
no open hand is seen within 30 seconds, the attempt is rejected with "no
open hand acquired in 30 s" and starts again.

READY is the 1 second between the two short low beeps and the GO beep,
counted from the end of the low beeps, so the GO beep never comes sooner
than it did before. On 2026-10-02 the two beeps came back to back and
sounded alike, and the lateral grasp was closed on the first one: by the
second there was nothing left to time. Now the two cues cannot be taken for
each other (low, quiet and double against high, loud and long), and the
recorder keeps watching the hand through READY. If the hand moves at 50 mm/s
or faster, which is the speed of a hand closing, you hear a low beep and the
window says "CLOSED TOO SOON - keep the hand open until the GO beep". If the
camera loses the open hand, it says "LOST YOU: show the open hand to the
camera again". Either way open the hand: the recorder goes back to OPEN
HAND and the attempt carries on. That is a reset, not a rejection.

MAKE THE GRASP gives you 4 seconds. If the camera loses your hand for more
than 0.3 seconds while you close it, you hear a low beep and the window says
"LOST YOU: keep the grasp, turn the palm to the camera". Keep the grasp and
bring the palm back toward the camera: the recorder waits for a hand it
treats as the same one (the same id, or a new id within 10 cm of where it
was lost: a rule of id and distance, not a proof of identity, so keep the
other hand out of view meanwhile), up to 4
seconds each time (`--lost-wait`) and never later than 8 seconds after the
GO beep (`--form-window`). Once it has the hand again, it holds it tracked
for 1 second, and then HOLD STILL starts as always (never before the 4
seconds are up). The take is then judged by the same acquisition gate as
every take. If the hand is not found again in time, the attempt is
rejected. At the end of the 4 seconds, if the hand was never lost, the
recorder checks that the hand did close after the GO beep: if nothing moved
at 50 mm/s or faster since then, the hand was already closed, and you get
CLOSED TOO SOON and go back to OPEN HAND (a reset). An attempt is reset at
most 10 times; the 11th close before the GO beep (or loss during READY)
rejects it. The limit only ends a loop that would not end otherwise.

These rules date from 2026-10-07. Until then a loss while forming sent you
back to OPEN HAND, and the third send-back of an attempt, a loss or an
early close, rejected it before its hold was ever recorded. The hold and
its gate did not change.

Each take's meta file says which rules it was recorded under,
`recorder_rules` ("2026-10-07"; takes recorded before have none), and
records how it went under `coaching`: `acquire_s` (seconds to find the open
hand), `ready_pause_s` (the READY seconds), `forming_lost_count` (how often
the hand was lost while you closed it and found again) with the height, palm
angle, seconds gone and when it was found of every loss in
`forming_losses`, `ready_losses` (lost during READY), `closed_before_cue`
(how often it closed before the GO beep) with how fast and when in
`early_closes`, `resets` (the returns to OPEN HAND), and the limits it ran
with (`lost_wait_s`, `form_window_s`, `relock_s`, `reset_limit`).
`--no-coach` brings back the old 5 second countdown without OPEN HAND,
unchanged.

The change into the grasp is recorded too, because the professor asked for
the time it takes to change from one pose to the next. Every camera frame
from the GO beep of MAKE THE GRASP until the take starts goes to
`forming\<take>.jsonl`, in the same format as the take's own file. After a
reset the next MAKE THE GRASP starts that file again, so it holds the try
that led to the take. A hand lost while you close it and found again stays
in the one file, with the gap: that file then has `"forming_lost_count": N`
on every line, and nothing is timed from it, because a change of pose that
the camera did not see cannot be timed (`transition_s` null with the note
"the hand was lost while forming the grasp (N time(s)): ..."; the
professor's folder marks its row invalid, with no duration, and the take's
note says so). From the file the recorder
measures `transition_s`: the seconds from the GO beep until your hand has
settled after its close. The close is the fastest movement after the beep
(`peak_speed_mm_s`, at `peak_at_s` seconds), and settled means the average
speed of the hand's 26 joints stayed under 20 mm/s for 0.3 s, searched for
after that peak, so a twitch on the beep and a pause before the real close
are not taken for the grasp. A still hand moves its joints at about 6 mm/s
in the camera's data, so 20 mm/s is still; a real close peaks at 180 mm/s
or more. `last_move_s` is the end of the hand's last movement at 50 mm/s or
faster. `transition_note` is empty for a plain close, and otherwise says:

- "no close after the cue (peak X mm/s): the hand was already in the
  grasp": nothing after the GO beep moved at 50 mm/s; `transition_s` is
  null. With READY this should no longer happen in a kept take.
- "settled, then moved again at T s": the hand settled after the close, then
  adjusted its grip from T s on; `transition_s` is the settle after the
  close, `last_move_s` the end of the adjustment.
- "never settled before the hold": no settled moment after the close before
  the take began; `transition_s` is null.
- "the hand was lost while forming the grasp (N time(s)): a change of pose
  is not timed across a gap in the tracking": the file is marked (above);
  `transition_s` and every other number are null.

The meta keeps all of these under `coaching` beside `forming_frames` (the
file's frame count) and `forming_tracked_fraction`, and `transition_rule`
the thresholds they were measured with. `transition_s` runs from the cue,
so it holds your reaction time as well as the movement; the professor's
folder times the movement alone, from its own start (see "The time to
change pose" at the end). Takes recorded before 2026-10-02
have no forming file. The six takes recorded on 2026-10-02 before this rule
were re-timed in place with step 8c.

Your other hand. You may hold your right hand open in view, palm to the
camera, at least 20 cm to the side of the module at about the same height,
if that helps the camera tell left from right. During MAKE THE GRASP, and
during a reacquisition after a loss (LOST YOU, until the hand is found
again), keep it out of the camera's view: the reacquisition rule is by id
and distance, not identity, and could adopt another hand that appears
within 10 cm of where yours was lost. OPEN HAND only takes a hand within
about 31 degrees of the module's axis (nearer the axis than 0.6 times its
height), so the hand off to the side is never the one acquired. The take
follows the hand OPEN HAND acquired by the tracker's id, including a new id
within 10 cm of where that hand was last seen, in a frame without it, which
the recorder treats as the same hand. The other hand's frames stay in the
take's file and are listed in the meta under `other_hand_ids`; they are
never measured.

## The COPY THIS window

During OPEN HAND the window says "OPEN HAND, palm to the camera" in large
letters, with what the camera sees of your hand and the seconds left. The
paper's picture is not shown yet: copying it before the camera has the hand
is what lost the hand on 2026-10-01. During READY it says "KEEP HAND OPEN"
and "close on the GO beep" in large letters, still without the picture.
After a reset (CLOSED TOO SOON, or LOST YOU during READY), the next OPEN
HAND says why in red above the large letters. At the GO beep the picture to
copy, cut from the paper's figure, appears large with the grasp's name, the
take number and MAKE THE GRASP with its countdown; the grasp's orientation
hint is under that line. If the hand is lost while you close it, the line
reads LOST YOU with the seconds left to find it, and "keep the grasp, turn
the palm to the camera" in red where the hint was. HOLD STILL and the
seconds left follow while the take
records. After the take the same window shows the review, the paper's
picture beside your still, and then the next grasp's OPEN HAND. The sheet
`grasp sheet.pdf` beside the papers is only a backup. If a picture is
missing, the window shows the grasp's shape in words instead and the
console says so.

## The orientation envelope

Every grasp is recorded inside this envelope (plan D3):

- the hand is 20 to 40 cm above the module;
- the wrist is inside the camera's view;
- every finger chain is visible to the lens;
- no finger is edge-on to the lens;
- a grasp that is naturally palm down (lateral key pinch, hook, extension
  plate) is turned only as far as the four points above need.

You do not have to write the rotation down. For every take the recorder
measures the palm's height and its angle to the lens (0 degrees is the palm
square to the lens, 90 is edge-on) and writes both into the take's meta
file.

## The orientation rule

The fingers make the paper's shape, but the forearm turns so the palm and
the fingertips face the camera. A vertical hand with the fingertips pointing
at the lens is what the camera cannot hold: on 2026-10-01 that is how the
tip, fingertip, tripod and lateral grasps were lost, while the same grasps
turned toward the camera tracked 99 to 100 %.

The ten grasps whose photo in the papers points the fingertips down carry
this hint in `protocols\grasps.json` (`orientation`): "turn the forearm so
the palm and the fingertips face the camera; same fingers, different angle".
They are p1_tip, p1_lateral, p1_palmar, p2_s04_fingertip_grasp,
p2_s05_fingertip_grasp_side_support, p2_s06_tripod_grasp,
p3_circular_precision_thumb_4_fingers, p3_circular_precision_thumb_2_fingers,
p3_circular_precision_thumb_1_finger and p3_prismatic_precision. Every other
grasp says "palm toward the camera". The console prints the hint once per
grasp, the COPY THIS window shows it during MAKE THE GRASP, and each take's
meta file keeps it as `orientation_hint`.

## The acquisition gate

Checked automatically after every take:

- at least 90 % of the take's frames tracked;
- the hand not lost inside the static interval. The static interval is the
  2 seconds of the take in which the hand moved least; the summary frame is
  chosen from it. Lost means gone from the data for more than 0.1 s before
  the tracker finds it again. The tracker also re-labels a hand it never
  stopped seeing (a new hand id with no hole in the data; seven of nine
  "losses" in the 60 s test of 2026-09-30 were that); those are counted in
  the review line and the meta but do not reject the take. A label flip
  (the tracker calling your left hand "right" under a new id) with no hole
  in the data is not a loss either: the take follows the hand, whatever it
  is called;
- the hand fitted as your hand. The static interval and the summary frame
  are chosen from the frames the tracker labelled as your hand, because a
  skeleton fitted as the other hand is a mirror image of yours. A take with
  no such frame, or with them in under half of the static interval, is
  rejected with "the tracker fitted the hand as a right hand for the whole
  take" or "... for most of the static interval".

The gate is the minimum for a usable take, not the decision. That is yours,
at the review.

## The review window

It shows the still, the grasp's name and these numbers:

- tracked %: the share of the take's frames in which the hand was tracked;
- hand: the tracker's id of the hand that was measured and its label on the
  summary frame, another hand in view (ignored) and how many frames the
  tracker called it the other hand;
- grab and pinch: the tracker's own 0 to 1 values for the summary frame;
- curls: one number per finger, the same curl every other report in this
  repo uses;
- palm height and angle to the lens: check the height is 20 to 40 cm.

Compare the still with the paper's figure, fingertip by fingertip. A take
can be 95 % tracked with one fingertip wrong. Then press one key:

| Key | What it does |
|-----|--------------|
| Enter or space | keeps the take |
| r | records the take again; this attempt goes to `rejected\` with the reason "operator redo" |
| q | ends the session; this attempt goes to `rejected\` |

If a key does nothing, click the COPY THIS window once and press it again. The
key and the time you pressed it are written to the meta file and to
session.json.

## The camera window

The live IR picture with the tracked skeleton drawn on it. Use it to put
your hand inside the envelope during OPEN HAND. It has a LEFT line and a
RIGHT line: that label is the tracker's guess, and it often calls the left
hand "right". Ignore the label. What matters is that one line says tracked
and its height reads OK. The other line saying NO HAND is expected, or
tracked when you hold the other hand in view.

## When a take is rejected

The console prints `REJECTED by the gate:` and the reason, which names the
losses behind it, for example `tracked 72 percent: lost 3 times, longest
1.4 s with the hand at 49 cm (too high: keep the palm 25 to 35 cm above the
module); the gate needs 90 percent`. Every loss is also listed in the take's
meta file under `gate.losses`, and `docs\tracking_quality.md` explains each
cause. The attempt's files move to `rejected\` beside a `<take>.reason.txt`, and the
recorder starts the same take again at OPEN HAND, up to 2 more times. Before
the next GO beep, fix the cause:

- tracked under 90 %: bring the hand inside 20 to 40 cm, keep the wrist in
  view, and turn the hand so no finger is edge-on;
- the hand was lost inside the static interval: hold the grasp still from
  the beep that starts the recording to the low beep and keep the hand
  over the middle of the module, 25 to 35 cm up;
- the tracker fitted the hand as a right hand: hold your right hand open in
  view, 20 cm or more to the side, but out of view while you close the left
  and while a lost hand is found again (Your other hand, above), and keep
  the left palm toward the camera while you close it.

Three rejections happen before anything is recorded, and the console prints
`REJECTED:` with the reason:

- `no open hand acquired in 30 s`: hold the open hand flat over the middle
  of the module, 18 to 40 cm up, palm toward the lens;
- `lost the hand while forming the grasp and it was not found again within
  the 4 s wait for it: at 21 cm with the palm 75 degrees from the lens (palm
  turned away: ...)` (or `... before the forming window ended, 8 s after the
  GO beep`): follow the orientation rule above, and when LOST YOU comes, turn
  the palm back to the camera without opening the hand;
- `closed before the GO beep 11 times; reset to OPEN HAND 10 times, the limit
  that ends a stuck loop; the last during READY, 0.3 s after the READY beep,
  at 96 mm/s`: keep the hand open through READY and close only on the GO
  beep. A reason that also says `lost the hand during READY N times` counts
  both kinds of reset.

Sessions recorded before 2026-10-07 also hold the earlier rules' reasons,
`lost the hand 3 times while forming the grasp, ...` and `closed before the
cue 3 times, ...`. The professor's README counts each kind apart.

After 3 failed attempts the recorder moves on to the next take. The end
table then lists that grasp as short of its takes, and step 8 below records
what is missing. The packager's README lists the grasps still short of 3
kept takes, with the most common reason they were rejected.

## Steps

### Step 1. No hardware: open PowerShell in the repo folder

```
cd "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
```

Every command below runs from this folder.

### Step 2. No hardware: rehearse the whole session on synthetic hands

```
.venv\Scripts\python.exe scripts\leap\record_poses.py --mock --protocol protocols\grasps.json --hand left --auto-accept --takes 1 --duration 1 --prep 0.5 --no-open
```

It takes about a minute and ends with a table in which every grasp says
yes. The rehearsal is written to `recordings\protocol_mock\grasps\`, never
beside the real sessions, and its stills say MOCK. Every take there also
has a file in `forming\`, and its meta a `transition_s` of about 0.15 s:
the synthetic hand closes in an eighth of a second.

### Step 3. No hardware: rehearse a lost hand

```
.venv\Scripts\python.exe scripts\leap\record_poses.py --mock --protocol protocols\grasps.json --hand left --items p1_tip --takes 1 --duration 2 --auto-accept --no-open --mock-lose-forming 1 --mock-back-after 0.5
```

The synthetic hand turns its palm edge-on and disappears while it closes,
the way the camera lost the tip grasp on 2026-10-01, and is back half a
second later, closed. The console prints `LOST YOU: keep the grasp, turn the
palm to the camera` once, then `found again after 0.5 s: hold the grasp
there 1 s before the take`, and keeps the take at its first attempt. That
take's meta file says `"forming_lost_count": 1`, and every line of its
forming file carries the same. It takes about 10 seconds. Without
`--mock-back-after 0.5` the hand stays gone: each attempt is rejected after
the 4 second wait, `lost the hand while forming the grasp and it was not
found again within the 4 s wait for it: ...`.

### Step 3b. No hardware: rehearse closing before the beep

```
.venv\Scripts\python.exe scripts\leap\record_poses.py --mock --protocol protocols\grasps.json --hand left --items p1_tip --takes 1 --duration 2 --auto-accept --no-open --mock-close-early 1
```

The synthetic hand closes on the two short READY beeps, during READY, the
way the lateral grasp was closed on 2026-10-02. The console prints `CLOSED
TOO SOON - keep the hand open until the GO beep` once and `reset 1 of 10:
closed before the GO beep, during READY, ...`, goes back to OPEN HAND, and
keeps the take on the next round of the same attempt. That take's meta file
says `"closed_before_cue": 1` and `"resets": 1`. It takes about 10 seconds.
With `--mock-close-early 11` the attempt is dropped at the 11th close, with
the reason `closed before the GO beep 11 times; reset to OPEN HAND 10 times,
the limit that ends a stuck loop; ...`.

### Step 4. Camera only, no hand over it: check the camera

```
.venv\Scripts\python.exe scripts\leap\check_setup.py
```

No line may say FAIL. Line 7 says WARN while no hand is over the camera,
which is fine. If a line says FAIL, do what that line says, then run this
step again.

### Step 5. Camera only, bare left hand: see the joint frames live

```
.venv\Scripts\python.exe scripts\joint_frames_view.py --live
```

Hold the left hand over the module. The window draws a small x y z triad
(x red, y green, z blue, 1 cm long) at each of the 26 joints and lists, on
the right, every joint's position in millimetres from the wrist and its
flexion and abduction, then the paper's 24 angles (Cobos et al. 2009). Close
a fist slowly: the PIP numbers climb toward 90. Press n to highlight one
finger at a time, s to save a picture of the window into
`recordings\joint_frames\`, q to close it. What the numbers mean:
`docs\joint_frames.md`. The window only reads the camera, so it can also
stay open while you record.

### Step 6. Camera only, bare left hand: run the 60 second tracking test

Go through the physical checklist at the top of `docs\tracking_quality.md`
first (module flat, lenses up and wiped, cable straight into the laptop, no
sunlight, one hand over the module). Then:

```
.venv\Scripts\python.exe scripts\leap\tracking_quality.py --hand left
```

Follow the instructions it prints and shows in the camera window for one
minute, slowly. It ends with a line that starts with `VERDICT:`. If the
verdict names a cause, apply its fix and run this step again. Go on to the
next step when the verdict says no losses, or when the only losses are the
ones you caused on purpose (50 cm, the far side, the palm turned away).

### Step 7. Camera only, bare left hand: record the grasps

Before you press Enter: the module flat on the table with the lenses up, no
sunlight on it, and your right hand resting on the table away from the
module, or held open at least 20 cm to the side of it (Your other hand,
above). Each take starts with OPEN HAND; the picture of each grasp appears
in the COPY THIS window at the GO beep.

```
.venv\Scripts\python.exe scripts\leap\record_poses.py --protocol protocols\grasps.json --hand left --operator "N Kim"
```

It waits until it sees a hand, then runs every grasp in the file, 3 takes
each. Plan on about 15 minutes of recording plus your time at the reviews.
At the end it prints the table (grasp, take, accepted, tracked %, grab,
pinch) and the folder, and opens the folder in Explorer.

### Step 8. Camera only, bare left hand: record the takes the table says are missing

Do this step only when a line under the table starts with
`Short of 3 kept takes`. Below that line, after "Record the missing takes
into this same session:", the recorder prints the exact command, which looks
like this:

```
.venv\Scripts\python.exe scripts\leap\record_poses.py --protocol protocols\grasps.json --hand left --resume "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer\recordings\protocol\grasps\20260928_140501_left"
```

Copy the printed command, not this example. It adds the missing takes to
the same session folder and carries the take numbers on, so Set A stays one
session (plan D9). A session recorded before the orientation hints were
added to `protocols\grasps.json` resumes too: the hints are the only change
it lets through, and session.json notes it under `protocol_changes`.

### Step 8b. Camera only, bare left hand: top every grasp up to 5 takes

The professor asked for 5 takes of every grasp. The session of 2026-10-01
has 3. This adds takes 4 and 5 of every grasp to that same session folder
and records nothing else:

```
.venv\Scripts\python.exe scripts\leap\record_poses.py --protocol protocols\grasps.json --hand left --operator "N Kim" --resume "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer\recordings\protocol\grasps\20261001_101621_left" --takes 5
```

`resume_set_a.cmd` on the Desktop runs exactly this command. A grasp that
already has 5 kept takes is skipped; one with fewer gets only the takes it
is missing. From then on session.json says 5 takes per grasp, so the end
table's "Short of 5 kept takes" and its printed command count to 5 too. The
new takes have forming files; the first three do not. The command is the
same under the rules of 2026-10-07 (above); the takes it adds from then on
are recorded under them and say so in their meta (`recorder_rules`), and
session.json notes when the session changed to them
(`recorder_rules_changes`). The takes already in the session keep their
meta as it was.

### Step 8c. No hardware: re-time the takes recorded before the READY pause

The six takes with forming files recorded on 2026-10-02 before READY and the
peak-anchored rule existed were timed by the first rule. This rewrites only
their transition numbers in the meta files (`transition_s`,
`peak_speed_mm_s`, `peak_at_s`, `last_move_s`, `transition_note`,
`transition_rule`) and leaves every other byte of the session as it was:

```
.venv\Scripts\python.exe scripts\leap\recompute_transition.py recordings\protocol\grasps\20261001_101621_left
```

It prints one line per take, the old `transition_s` and the new one, and
ends with `recompute ok`. Running it again changes nothing and says
`(unchanged)` on every line. Add `--dry-run` to see the lines without
writing anything. `check_protocol.py` notes "recompute needed" on a take
whose meta still lacks these keys; it is not a failure.

### Step 8d. Transitions (grasp to grasp). No hardware: rehearse them

The forming files time one change only, the open hand into the grasp. A
change from one grasp into another is recorded by
`scripts\leap\record_transitions.py` (steps 8d to 8f): you hold grasp A
still over the camera, a high beep sounds after a random wait, you change
to grasp B and hold it still until the next instruction. A take is kept
when its timing is valid or questionable, and asked for again otherwise
(twice at most).

Double-click `rehearse_transitions.cmd` on the Desktop. It runs:

```
.venv\Scripts\python.exe scripts\leap\record_transitions.py --mock --hand left --pairs "open_hand>p1_cylindrical,p1_cylindrical>p1_spherical" --takes 2 --operator "N Kim"
```

A synthetic hand does the changes: two pairs, two takes each, about half a
minute. It ends with `kept 4 of 4 take(s) in
...\recordings\protocol_mock\transitions\<date>_<time>_left`. The rebuild
never takes this folder.

### Step 8e. Transitions (grasp to grasp). Camera only, bare left hand: one change as a dry run

In PowerShell:

```
& "C:\Users\nkim2\OneDrive\Desktop\record_transitions_left.cmd" '"open_hand>p1_cylindrical"'
```

Type the pair in single quotes around double quotes, as shown: Windows
PowerShell drops plain double quotes, and the launcher then reads `>` as
"write into a file" and records nothing. (Double-clicking the launcher
records nothing either: it prints this command and the one of step 8f.)
Each take: hold the open hand still, palm to the camera, 25 to 35 cm above
the module; on the high beep form the cylindrical grasp and hold it. The
launcher records 3 kept takes and ends with `kept 3 of 3 take(s) in
...\recordings\protocol\transitions\<date>_<time>_left`. These are real
recordings of the open hand into the cylindrical grasp, and the rebuild
puts them into the table with the others.

### Step 8f. Transitions (grasp to grasp). Camera only, bare left hand: the professor's pairs

The same launcher with the pairs the professor asks for: each pair
`<from>><to>` with the grasp ids of `protocols\grasps.json`, the pairs
separated by commas, the whole list in single quotes around double quotes.
For example:

```
& "C:\Users\nkim2\OneDrive\Desktop\record_transitions_left.cmd" '"p1_cylindrical>p1_spherical,p1_spherical>p1_tip"'
```

It records 3 kept takes of every pair. Each run is one folder,
`recordings\protocol\transitions\<YYYYMMDD_HHMMSS>_left\`: `session.json`
(rewritten after every attempt), `pose_transitions.csv` (one row per
attempt), `leap\<from>_to_<to>_left_take<N>_<time>.jsonl` (every frame from
1 s before the beep to the end of the hold) and `meta\` (the beep, the hand
followed, its timing). An attempt that was not kept stays there with its
reason. Then run step 10 as it is: it prints `transition session <folder>:
<N> kept take(s)`, puts the kept takes into `transitions\` of the
professor's folder, and its `TRANSITION TIMING` line counts how many of
them are between two grasp types.

A duration from these recordings is quoted with its thresholds, never
alone: the movement starts at the first frame of 0.1 s or longer of pose
speed at or above 25 mm/s, and is settled once the pose speed stays below
10 mm/s for 0.3 s (every row's `movement_start_method` and
`settled_method`). How much the reading moves with them is in "The time to
change pose" below.

### Step 9. No hardware: look at the session folder

```
explorer recordings\protocol\grasps
```

Open the newest folder. Every grasp should have 3 files in `leap\`,
`stills\`, `keypoints\` and `meta\` (5 after step 8b), and every take
recorded since 2026-10-02 one in `forming\`. The stills themselves stay
in the session folder and never go into the professor's folder as they
are: the camera looks up past the hand, so most of them show your face.
What goes in is a cut of each: the rebuild (step 10) cuts every still that
has your hand's skeleton drawn on it to that hand.

### Step 10. No hardware: rebuild the professor's folder

Double-click `rebuild_professor_grasp_set.cmd` on the Desktop. It runs:

```
.venv\Scripts\python.exe scripts\rebuild_grasp_set.py --session recordings\protocol\grasps\20261001_101621_left --out "C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set for professor 2026-10-01" --operator "N Kim"
```

Run it again after every recording session. It does three things, and
prints `1/3`, `2/3`, `3/3` as it goes:

1. The joint frames: for every kept take, the 26 joints of its summary
   frame (position, axes, angles) to `joint_frames\<take>.csv` inside the
   session folder, a drawing to `joint_frames\<take>.png`, and one
   `joint_frames\joint_frames.pdf` with a page per take.
2. The folder: the kept recordings and their keypoints, each grasp's
   frame-by-frame file, `grasps_summary.csv`, `grasp_types_audit.csv`,
   `joint_angles_24.csv`, the joint-frame CSVs and the PDF (never the
   PNGs), the joint key (`grasps\joint_frames\joint_key.csv` and
   `joint_key.pdf`, `docs\joint_frames.md`), each kept take's still cut to
   the recorded hand (`grasps\images\<grasp>_left_take<N>.png`) with
   `grasps\images\grasp_images.pdf` (one page per grasp type),
   `transitions\pose_transitions.csv` with the forming files, and
   `README.txt`.
3. The check: the finished folder is read back and measured again, and the
   report is printed and written to `validation_report.txt`.

The command needs no change for transition sessions (steps 8e and 8f).
Under `2/3` it looks in `recordings\protocol\transitions`
(`--transitions-dir`, by default the `transitions` folder beside the
session's own `grasps` folder) and adds every session of the transition
recorder there to `transitions\pose_transitions.csv`, in name order. It prints
`transition session <folder>: <N> kept take(s)` for each one, and
`no transition sessions in ...` when there is none. A rehearsal session
(`--mock`) is never in that folder; if one is put there, it is skipped with
a line that says so (`--allow-mock` takes it, for rehearsing only). A
folder named with `--transitions` is added as well, and none twice.

The last four lines it prints are the ones to read:

```
STATIC-POSE TASK: COMPLETE (...)
FULL MULTI-TAKE DATA COLLECTION: INCOMPLETE (69 of 115 recordings against our replication target; 46 still to record)
TRANSITION TIMING: PARTIAL (2 valid and 2 questionable of 6 recording(s) of a change of pose; none between two grasp types)

FOLDER CHECK: PASSED
```

The first is the task as the professor set it: one labelled static pose for
every grasp type. The second is our replication target of 5 takes per grasp. The
third says whether any recorded change of pose has a valid timing; it reads
PARTIAL or MISSING and never more, because no number of them was asked for.
The fourth is the folder itself: `FAILED` means a file in it is wrong or
missing, and the lines above it say which. Among the counts above them,
`pictures cut to the hand: PNGs that open and are not black`,
`recordings without a picture`, `grasp_images.pdf pages` and
`joint key rows` are the pictures and the key.

Then look at the pictures, every time a new session went into the folder,
before the folder is handed in: open `grasps\images\grasp_images.pdf` in
the professor's folder and check every picture for a face or anything of
the room. The cut keeps every pixel within 0.3 hand radii of the drawn
skeleton and, out to 0.5 radii, those at least 0.7 times as bright as the
hand (the whole rule is in README section 5 and
`src\leap_hand\grasp_images.py`). It is a cut
by distance and brightness, not a face detector: its numbers were set on
the 69 stills of `20261001_101621_left`, every result looked at on
2026-10-06, and a new session's stills can differ. A take with no still,
or with no skeleton of your hand drawn on it, gets no picture; README
section 5 names it.

Those pictures are the only images the folder may hold. If the build
would put any other image in it, or the folder already holds one (by its
name or its first bytes), the build stops with `REFUSED` and exit code 1,
names the file and writes nothing.

What the README says the pictures show in or against the hand comes from
`protocols\grasps_still_notes.json`: read by eye on 2026-10-06, not
confirmed by the operator, and for the session it names only. For any
other session the build ignores it and the README says that the pictures
were not read for it.

The recordings in the session folder are only read. A file of the hand-in
folder that the rebuild changes, or no longer makes, is moved to
`xr trainer\_superseded\grasp set for professor 2026-10-01\<time>\` first;
nothing is deleted. Running it twice in a row changes nothing the second
time.

Durations in the summary come from the timestamps in each recording, never
from the number of lines: a take recorded with your other hand in view has
two lines per camera frame.

### Step 11. No hardware: check a folder without rebuilding it

```
.venv\Scripts\python.exe scripts\validate_grasp_set.py "C:\Users\nkim2\OneDrive\Desktop\xr trainer\grasp set for professor 2026-10-01"
```

It prints the same report as step 10's third part and changes nothing.

## The time to change pose

Two different things are called this, and only one is recorded so far.

Open hand into a grasp. Every coached take records it: the forming file
(above). In the professor's folder each forming file becomes a row of
`transitions\pose_transitions.csv`, timed from the start of the change of
pose: `transition_duration_s` = settled minus movement start, with
`reaction_time_s` (cue to movement start) beside it. The forming file of a
take kept after the hand was lost while forming (since 2026-10-07) is not
timed: its row is invalid, "the hand was lost while forming the grasp", with
no duration, and README section 17 counts it among the invalid.

What is timed is the pose, not the hand over the camera. The rule reads the
pose speed: the joints' speed after the whole hand's movement has been
taken out (each frame laid on the earlier one by the rigid fit of the
palm's joints). The movement is the stretch of pose speed at 25 mm/s or
faster, 0.1 s or longer, that holds the fastest frame after the cue;
settled is the first frame after its fastest from which the pose speed
stays under 10 mm/s for 0.3 s. A pose that was already changing at the cue
gets no start and no duration.

A first version, the same day, read the speed over the camera, and an
independent review showed what that does: it passed one take of
2026-10-02 (p1_cylindrical take 4) and gave it 1.50 s, most of which was
the open hand being carried 25 mm sideways and back. On the pose speed the
six forming files of that day read: four changes of pose, starting 0.57 to
0.81 s after the cue and lasting 0.80 to 1.11 s (two valid, two
questionable), and two takes whose hand was already in the grasp at the
cue. In three of the four the hand as a whole was being moved over the
camera at the cue with its pose still, which the READY step now prevents.

Three cautions, all stated in the professor's README. The closes are
coached and unhurried, and one of the four is a close the recorder had just
asked to be made slower (`lost_while_forming` in its meta). The length
depends on the thresholds by a factor of three: at twice the thresholds
(50 and 20 mm/s) the same four read 0.30 to 0.46 s, the fast part of the
close alone, and the README prints both that and the reading at half. And
the reading depends on the size of the change: on a synthetic hand whose
fingers move for exactly 0.50 s it reads 0.34 s for a change of 15 mm RMS,
0.43 s at 30 mm, 0.49 s at 61 mm and 0.51 s at 122 mm, and does not see a
change of 8 mm at all (`leap_hand.transition.known_move_reading`);
`pose_change_mm` in the table is that size for every row.

A `--mock` rehearsal closes the synthetic hand at the instant of the cue,
with no reaction time, so its rows read "already changing at the cue".
That is the rule working, not a fault in the rehearsal.

One grasp into another. Not in the protocol; step 8f records it (camera
only, bare hand): you hold grasp A still, a high beep sounds after a random
wait, you change to grasp B and hold it. As of 2026-10-07 it has been run
on synthetic hands only (`rehearse_transitions.cmd`); no take has been
recorded with the camera.

The static recordings are never timed: not from the time between two
files, not from the gaps between takes.
