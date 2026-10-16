# Runbook: recording Set B (finger flexion) and Set C (sequences)

The professor asked for single-finger flexion (Set B) and seven sequential
movements (Set C), recorded with the gloves. `scripts\record_protocol.py`
records both, one hand per run. Since 2026-10-12 they are recorded with the
gloves alone (`--camera none` in every command below, and in every Desktop
launcher of these sets): the professor's instruction for these two sets is
the data glove, and the Leap Motion camera is for the grasp set only. It is
hands-free once it starts: every movement is called by a beep, shown in
words on a small window and printed in the console. The plan behind it is
`docs\grasp_and_flexion_protocol_plan.md`; the files it writes are specified
in `docs\protocol_formats.md`. Running the camera beside the gloves is still
possible and is described at the end ("With the camera beside the gloves").

Every command below is typed in PowerShell in the repo folder
(`C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer`).

## Part 1: rehearse with no hardware

A rehearsal uses a mock glove and no camera, so its folders go under
`recordings\protocol_mock\` and its `session.json` says `"mock": true`:
rehearsals never sit beside real sessions, and the packager refuses them.

Step 1. **No hardware.** Rehearse Set B with the mock glove (about 25
seconds):

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set finger_flexion --hand left --camera none --items index_fast --mock-glove --no-view --no-open --calibrated-at 2026-09-28T14:00:00 --time-scale 0.25
```

The console ends with `1 accepted, 0 rejected` and the folder it wrote,
under `recordings\protocol_mock\finger_flexion\`.

Step 2. **No hardware.** Rehearse Set C the same way (about 25 seconds):

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set sequences --hand left --camera none --items seq3_alternating --takes 1 --mock-glove --no-view --no-open --calibrated-at 2026-09-28T14:00:00 --time-scale 0.25
```

The console ends with `1 accepted, 0 rejected` and the folder it wrote,
under `recordings\protocol_mock\sequences\`.

Step 3. **No hardware.** Rehearse a rejected take. This mock glove does the
warm-up and then keeps the hand open, so the take must be rejected:

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set finger_flexion --hand left --camera none --items index_fast --mock-glove-ignore-cues --no-view --no-open --calibrated-at 2026-09-28T14:00:00 --time-scale 0.25 --retries 0
```

The console says `REJECTED: the index moved only 0.0... of its warm-up range
(needs 0.60)`, and the attempt's files are in its folder under
`recordings\protocol_mock\finger_flexion\`, inside `rejected\`, with the
reason in `rejected\<take>.reason.txt`.

The other rejection worth hearing once is the speed rule's (see "What makes
a take accepted"). **No hardware.** This mock glove follows every cue but
bends at once, as a hand that snaps the finger shut on the beep does:

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set finger_flexion --hand left --camera none --items ring --mock-glove-snap --no-view --no-open --calibrated-at 2026-09-28T14:00:00 --time-scale 0.25 --retries 0
```

The console says `REJECTED: the ring bent too fast on 5 of 5 bends: 0.0... s,
and a 1 s bend cue needs at least 0.15 s ...` (a quarter of the real 4 s and
0.60 s), then `what to do: start on the high beep and keep the ring moving
for the whole 1 s, so it is fully bent only as the hold beep sounds`.

Step 4. **No hardware.** Check the folder from step 1 with the checker. Use
the folder path the console printed at the end of step 1:

```powershell
.venv\Scripts\python.exe scripts\check_protocol.py recordings\protocol_mock\finger_flexion\<folder from step 1>
```

It ends with `Every accepted take passes.`

## Part 2: the left hand

Step 5. **Gloves on, no camera.** Set up: both gloves charged and on; XR
Trainer running and streaming to port 9002. Close any other recorder that
listens on port 9002. The camera is not used; it can stay unplugged.

Putting the gloves on (StretchSense's advice of 5 October 2026, against the
readings that drift toward an open hand): pull each finger sleeve down until
its gusset reaches the webbing between the fingers, then wiggle and bend the
fingers for a few seconds so the glove settles, and only then calibrate. In
any calibration or warm-up that asks for a closed hand, close it fully but do
not squeeze: a squeezed fist deforms the hand in a way ordinary movement does
not. The launchers print the same reminder.

Step 6. **Gloves on, no camera.** Record Set B with the left hand (7 takes,
about 7 minutes of recording):

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set finger_flexion --hand left --camera none
```

What happens, in order:

1. The console asks you to calibrate both gloves in XR Trainer. Do it, then
   press Enter. The time you press Enter is saved as the calibration time.
2. The recorder waits for the left glove (its packets on port 9002). The
   right hand rests on the table for the whole run. Hold the left hand
   where it is comfortable, for example the elbow on the desk and the hand
   upright: the glove senses no position, only how far each finger bends.
3. Warm-up (about 30 seconds): a beep and "OPEN PALM" for 3 seconds, a
   higher beep and "FULL FIST" for 3 seconds, a low beep to open the hand,
   then each finger on its own, thumb first: a high beep and "bend the
   THUMB only" for 3 seconds (fold it fully and hold it there), a low beep
   and "straighten" for 1.5 seconds, then the index, the middle, the ring
   and the little finger the same way. This measures each finger's own
   range for today: a cued finger is measured against how far it bent on
   its own. The warm-up is refused, and done again (the console says
   `warm-up again (2 of 3)`), when a finger barely bent on its own; the
   reason says which finger and what to do, for example `in the
   single-finger warm-up the thumb bent only 12 degrees (needs 35): fold it
   fully across the palm, tip to the base of the little finger`. After
   three refused warm-ups the session is refused.
4. The takes: thumb, index, middle, ring, little finger, index slow, index
   fast. Each one shows its name for 3 seconds, then calls every bend, hold,
   straighten and rest with a beep. Move one finger only and keep the others
   straight. Follow the beeps; the bars on the cue window show what the
   glove measured (see "What you see and hear"). **A bend or a straighten
   that gets 4 or 6 seconds is a slow movement**: start on the beep and
   keep the finger moving for the whole countdown, so it arrives as the
   next beep sounds. Do not close it at once and wait. The small line under
   the cue says so (`cycle 2/5: slowly, use the whole 4 s`), and the
   finger's bar is white while it is under way. Only the fast index (1 s)
   is meant to be quick.
5. After each take the console says ACCEPTED or REJECTED with the reason. A
   take is rejected when the cued finger did not move far enough, did not
   bend fully on every bend beep (for example `REJECTED: the thumb bent
   fully only 1 time of 5 ...`), or bent too fast (`REJECTED: the index bent
   too fast on 5 of 5 bends: 0.05 to 0.13 s, and a 4 s bend cue needs at
   least 0.60 s ...`); the line under it then says what to do, for example
   `what to do: fold the thumb fully across the palm on every bend, then
   open it fully`. Every take, accepted or not, also gets a `speed:` line
   with how long the glove timed its bends and its straightens. A rejected
   take is recorded again straight away, up to two more times. Press `s` to
   give up on an item for this run (in the pause after a take, or during
   the take to end it at once).
6. At the end the console prints a table of every take and the folder, and
   the folder opens in File Explorer. When the run stopped before the end
   (Ctrl+C, `q`, `s`, a refused warm-up), the console also prints how to
   carry on in the same folder; see "Carrying on a session that stopped".

Step 7. **Gloves on, no camera.** Check Set B for the left hand. Use the
folder path printed at the end of step 6:

```powershell
.venv\Scripts\python.exe scripts\check_protocol.py recordings\protocol\finger_flexion\<folder from step 6>
```

Step 8. **Gloves on, no camera.** Only when step 6 or step 7 named a failed
take: record those items again, with their ids from the table, separated by
commas (for example `ring,index_fast`):

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set finger_flexion --hand left --camera none --items <failed ids>
```

Step 9. **Gloves on, no camera.** Record Set C with the left hand (21 takes:
the seven sequences in three rounds, each round in a shuffled order, about
10 minutes of recording):

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set sequences --hand left --camera none
```

The same start as step 6 (calibration, Enter, warm-up). Then every
step of a sequence is one beep and its words: make the shape, then hold it
until the next beep. A high beep means a finger flexes, a low beep means
fingers only straighten.

Step 10. **Gloves on, no camera.** Check Set C for the left hand. Use the
folder path printed at the end of step 9:

```powershell
.venv\Scripts\python.exe scripts\check_protocol.py recordings\protocol\sequences\<folder from step 9>
```

Step 11. **Gloves on, no camera.** Only when step 9 or step 10 named a failed
take: record one more take of each of those sequences:

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set sequences --hand left --camera none --items <failed ids> --takes 1
```

## Part 3: the right hand

Step 12. **Gloves on, no camera.** Record Set B with the right hand. The left
hand now rests on the table:

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set finger_flexion --hand right --camera none
```

Step 13. **Gloves on, no camera.** Check Set B for the right hand:

```powershell
.venv\Scripts\python.exe scripts\check_protocol.py recordings\protocol\finger_flexion\<folder from step 12>
```

Step 14. **Gloves on, no camera.** Only when step 12 or step 13 named a
failed take: record those items again:

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set finger_flexion --hand right --camera none --items <failed ids>
```

Step 15. **Gloves on, no camera.** Record Set C with the right hand:

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set sequences --hand right --camera none
```

Step 16. **Gloves on, no camera.** Check Set C for the right hand. This is the
second hand, so the bands come from this session's own open-hand and
full-fist steps (plan, section 4):

```powershell
.venv\Scripts\python.exe scripts\check_protocol.py recordings\protocol\sequences\<folder from step 15> --bands auto
```

Step 17. **Gloves on, no camera.** Only when step 15 or step 16 named a
failed take: record one more take of each of those sequences:

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set sequences --hand right --camera none --items <failed ids> --takes 1
```

Step 18. **No hardware.** Write down every folder path the console printed
in part 2 and part 3. The hand-in folder is built from them by
`scripts\package_professor_set.py` (its `--flexion` and `--sequences`
options take several folders each), for example:

```powershell
.venv\Scripts\python.exe scripts\package_professor_set.py --out "..\xr trainer\glove flexion and sequences <date>" --flexion <Set B left folder> <Set B right folder> --sequences <Set C left folder> <Set C right folder> --operator "N Kim"
```

Beside each take's files it writes the take's joint angles frame by frame
(`<hand>\joint_angles\<take>.csv`: every glove frame with the cue in force
and the 24 named angles in degrees) and, per hand, `cue_angles.csv` (the 24
angles at every held cue: each hold and rest of Set B, each step of Set C).
Those two are what the joint-limit and inter-joint-dependency work reads;
`docs\protocol_formats.md` section 9 has their columns. The README the
packager writes says the recordings are glove only and which nine of the 24
angles are the glove's hand model and not measurements. Read its "One value
per digit" lines before sending the folder: on this glove the three flexions
of a digit are one measurement spread over its joints by XR Trainer's hand
model, so the files show how far one finger goes along with another, not how
the joints of one finger depend on each other.

An item of the protocol that a hand has no accepted take of is named in the
README whether or not anything is said about it: the Set B item table says
"not recorded" beside it, SETUP names it, and each session's lines say what
became of its attempts at it (rejected, stopped or skipped by the operator,
or not tried) and how the session ended. When an item was left out on
purpose, say why with `--not-recorded-reason`; the README prints it after
those facts. On 2026-10-15 the slow and fast index were left out of Set B
on both hands:

```powershell
.venv\Scripts\python.exe scripts\package_professor_set.py --out "..\xr trainer\glove flexion and sequences 2026-10-15" --flexion recordings\protocol\finger_flexion\20261015_204941_left recordings\protocol\finger_flexion\20261015_210752_right --sequences recordings\protocol\sequences\20261015_212038_left recordings\protocol\sequences\20261015_214026_right --operator "N Kim" --not-recorded-reason "the operator chose to record the five items at the 4 s pace only, not the slow (6 s) or the fast (1 s) index"
```

## A video of each take

**No hardware.** Nothing films the hand in a glove-only session. To see a
take, draw it from its glove data (`scripts\render_glove_video.py`,
`cam_hand.glove_video`): one MP4 per accepted take, in real time, with the
hand at its palm and from its side, the cue in force in words and each
digit's flexion angles. Use the folder path the recorder printed:

```powershell
.venv\Scripts\python.exe scripts\render_glove_video.py recordings\protocol\finger_flexion\<folder> --out "..\xr trainer\glove animations\<name>"
```

Every picture says along its bottom edge that it is drawn from the glove's
joint data and is not camera footage: the hand never moves or turns and the
fingers never spread, because the glove senses flexion only. The videos do
not go into the hand-in folder (the packager refuses any image or video
there); send them beside it.

## What you see and hear

One window. The cue window shows the take name, the
words of the current cue, a countdown and, along its bottom third, five
bars, thumb to little finger: each finger's glove reading as a fraction of
its range from today's warm-up, 0 at the bottom (open) and 1 at the top
(bent as far as in the warm-up). The two thin ticks beside every bar mark
0.3 and 0.6. The cued finger is drawn wide:

| Bar | Meaning |
|-----|---------|
| green | the glove reads the cued finger where the cue wants it: at or above 0.6 to bend, hold or flex, at or below 0.3 to straighten, rest or extend |
| amber | it does not (yet): bend further, or open further |
| white | nothing to judge: the pause after a take, or a slow bend or straighten under way (Set B, a 4 s or 6 s cue), where the finger has nowhere to be yet and should keep moving until the next beep |

Under the cue's words a small line says where the take is (`cycle 2/5`),
and during a slow bend or straighten how to make it (`cycle 2/5: slowly,
use the whole 4 s`). In a session recorded without the speed rule
(`--no-speed-check`, or one started before the rule was added on 2026-10-13
and carried on) the bar is green or amber through those cues too, as it
was.
| grey, narrow | a finger the cue does not name |

The bars show the glove against its own warm-up of today. `--no-bars` keeps
the cue window to the words and the countdown.

| Beep | Meaning |
|------|---------|
| high (1200 Hz) | bend, or a step where a finger flexes |
| middle (1000 Hz) | hold, or a step that changes nothing |
| low (800 Hz) | straighten, or a step where fingers only straighten |
| lowest (650 Hz) | rest (Set B) |
| short 500 Hz | the item is about to start |
| long 500 Hz | the take is over |

For 3 seconds after each take:

| Key | What it does |
|-----|--------------|
| `r` | redo this take: it moves to `rejected\` with "redo asked by the operator" |
| `s` | skip the rest of this item in this run: no more attempts at it (the take just recorded is kept if it was accepted) |
| `q` | stop the session after this take; everything recorded so far is kept |

During a take, `s` ends the take at once: it is moved to `rejected\` with
"skipped by the operator (s) during the take" and the item is skipped. A
skipped item is listed in the table at the end and in `session.json` under
`skipped`; a run that skipped an item with no accepted take ends with exit
code 1, like a failed take. The keys work on the cue window and in the
console. Ctrl+C stops at once; the take in progress is moved to
`rejected\` as interrupted.

## What makes a take accepted

- Set B: the cued finger has to move at least 0.60 of its own range from
  the warm-up (open palm to the finger bent on its own), and it has to bend
  fully once for every bend beep. A bend counts when the reading rises
  above 60 percent of the take's own range after being below 30 percent,
  the same count the checker (`scripts\check_protocol.py`) fails a take
  on, so a finger that folds fully on only some cycles is rejected on the
  spot instead of after the session. The other four fingers are measured
  (against the fist) and reported, never failed on.
- Set B, bend speed (since 2026-10-13): a bend the cue gives 2 seconds or
  more for has to be slow. Every counted bend is timed from 20 percent to
  80 percent of its own rise (from the lowest reading since the finger was
  last open to the highest before it opens again), and it must take at
  least 15 percent of the cue's time: 0.60 s of the 4 s bends, 0.90 s of
  the slow index's 6 s. One bend under that rejects the take. The fast
  index (1 s) is not judged. The straighten is timed the same way down and
  printed, never judged: the glove's reading falls quickly once a finger
  starts to open, however slowly the finger moves. The bar is set between
  what the recordings show (on 2026-10-13 a whole session's bends took
  0.05 to 0.55 s, at most 0.11 of their cue; the bends of a slow sweep
  paced at 5 s read 1.2 to 3.3 s on the glove, 0.24 of the cue or more),
  and it rests on that one sweep of one finger. If a finger that did move
  for the whole cue is still timed as a jump, record the session with
  `--no-speed-check`: the times are then reported, in the session and in
  the hand-in, and not judged. A session keeps the rule it was started
  with (`speed_check` in its session.json).
- Set C: in the last 1.5 seconds of every step, a finger the step says is
  flexed has to read above 0.60 of its range, and a finger the step says is
  straight must not read above 0.60. A straight finger between 0.30 and 0.60
  is reported as coupling (the ring pulls its neighbours along) and does not
  reject the take.
- The glove's thumb is measured by its joint angles in degrees (its three
  flexion angles summed), not by the fingertip distance, which barely moves
  when the thumb bends. The other four fingers keep the fingertip distance.
- The warm-up is refused when the glove's index, middle, ring or little
  finger moves less than 0.30 between the open palm and the fist, or when a
  finger bent on its own moves less than half of its fist range (the thumb:
  less than 35 degrees). It is done again up to three times before the
  session is refused.

## When something stops the run

| The console says | Do this |
|------------------|---------|
| `Only 0 glove packets ...` | Start streaming in XR Trainer, then run the same command again. |
| `The glove is streaming right, not left` | Put the glove on the hand named in the command, then run it again. |
| `it ran with --camera leap` (after `--resume`) | The folder is a session recorded with the camera before 2026-10-12. Start a new session: the same command without `--resume latest`. |
| `REFUSED: warm-up: the glove barely moved ...` | Calibrate both gloves in XR Trainer again, then run it again and make a full fist. |
| `REFUSED: warm-up: in the single-finger warm-up the thumb bent only ...` | Run the same command with `--resume latest` and, at "bend the THUMB only", fold the thumb fully across the palm and hold it until the low beep. |
| `REJECTED: the index bent too fast on ...` | Nothing to run: the take is recorded again by itself. Start on the high beep and keep the finger moving for the whole countdown, so it is fully bent only as the hold beep sounds. |
| `FAILED: index take 1 after 3 attempt(s)` followed by `if the finger did move for the whole cue ...` | The finger was timed as a jump three times. If it was moving for the whole cue each time, the glove is the cause: start a new session with the same command and `--no-speed-check` added, and say so when handing in. Otherwise run the same command with `--resume latest` and slow the bend down. |

## Carrying on a session that stopped

**Gloves on, no camera.** A session that stopped before the end (Ctrl+C, `q`,
`s`, a refused warm-up, a crash) carries on in its own folder. Run the same
command as before with `--resume latest` added, for example after step 6:

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set finger_flexion --hand left --camera none --resume latest
```

`latest` is the newest folder of this set and hand. To name one, give its
path instead of `latest`. What happens: the calibration prompt again
(Enter), the wait for the glove, and a new warm-up,
saved as `warmup_<HHMMSS>.json` beside the first one (the gloves came off
since, so the ranges are measured again; the first `warmup.json` is never
rewritten). Then the session's own round order goes on: items that already
have their accepted takes are passed over, and the rest are recorded,
numbered on from the accepted takes. Do not add `--items`, `--takes` or
`--seed`: the session already fixes them, and the recorder refuses them.
It also refuses a folder of another set or hand, a mock session in a real
run, a session recorded with the camera (every Set B and Set C session
before 2026-10-12: a session is glove only or glove and camera from its
first take to its last) and a session recorded before the glove thumb was
measured in degrees (before 2026-10-01); start a new session for those. An
item skipped with `s` is recorded again on resume.

The bend-speed rule is the session's too. A Set B session started under it
is carried on under it (adding `--no-speed-check` to the resume is
refused), and one started without it, with `--no-speed-check` or before
the rule was added on 2026-10-13, is carried on without it: the console
then says `Bend speed: timed and reported, not judged (the session was
started without the rule)`. To record under the rule, start a new session.

## Where the files go

One folder per run:
`recordings\protocol\<finger_flexion or sequences>\<date>_<time>_<hand>\`

| File | What it is |
|------|------------|
| `session.json` | the run: protocol file and version, hand, calibration time, round order and seed, every take with its verdict and the warm-up it was judged against, skipped items, resumed runs (rewritten after every take) |
| `warmup.json` | each finger's open, fist, span and single-finger bend for this run and the units (glove thumb in degrees); its `camera` block is null in a glove-only session |
| `warmup_<HHMMSS>.json` | the warm-up of each later run of a resumed session |
| `glove\<take>.jsonl` | the glove at its full rate, every line tagged with session, item and take |
| `leap\<take>.jsonl` | only in a session that ran the camera: the camera beside the glove, both hands if it saw two |
| `events\<take>.events.jsonl` | every cue at the moment of its beep, and every decision |
| `stills\<take>.png` | only in a session that ran the camera: one hand-cropped IR still per take; it stays in the repo, and neither it nor a cut of it goes into the hand-in: the folder's only pictures are the grasp set's stills cut to the hand (`docs\grasp_recording.md`, step 10), and the packager refuses any other image |
| `rejected\` | every attempt that was not accepted, same layout, with `<take>.reason.txt` |

`--time-scale` (used in part 1) shortens every duration for a rehearsal and
is saved in `session.json`; a real session always runs at the default of 1.
A run with a mock sensor writes the same layout under
`recordings\protocol_mock\` instead, marked `"mock": true`.

`session.json` says which sensors ran: `"camera": "none"` for a glove-only
session, `"camera": "leap"` for one with the camera beside the gloves. The
checker and the packager read that key and say only what the session's
sensors measured.

## With the camera beside the gloves

Not the professor's protocol for these sets; kept for comparing the glove
against the camera. **Gloves on + camera.** Leave `--camera none` out of a
record command and the Ultraleap runs beside the gloves as a reference: it
has to be flat on the desk, lens up, on USB, with the Ultraleap Tracking
service running. After the wait for the glove the recorder then waits for
the camera (hold the hand open, palm toward the lens, 18 to 28 cm above the
module; `REFUSED: no acquirable left hand in 60 s` means it never saw that,
`No live tracking: ...` that the camera or its service is missing), a
camera window shows the IR image and the skeleton (`--hide-camera` runs
without it), and each take also writes `leap\<take>.jsonl` and one IR
still. `--mock-leap` is the mock camera for rehearsing this without
hardware. The checker then also reports the camera's fractions, and for
Set B the transfer curve, hysteresis and lag where the camera followed the
finger. The Desktop launchers do not do this.

## The Desktop launchers

Each launcher on the Desktop goes to the repo folder, prints its hardware
setup, runs one command, and waits for a key at the end.

| Launcher | Setup | What it runs |
|----------|-------|--------------|
| `rehearse_set_b.cmd` | No hardware (mock glove, no camera) | step 1 without `--no-view`: Set B, index fast, at a quarter of the time |
| `rehearse_set_c.cmd` | No hardware (mock glove, no camera) | step 2 without `--no-view`: Set C, sequence 3, one take |
| `record_set_b_left.cmd`, `record_set_b_right.cmd` | Gloves on, no camera | Set B with that hand (steps 6 and 12) |
| `record_set_c_left.cmd`, `record_set_c_right.cmd` | Gloves on, no camera | Set C with that hand (steps 9 and 15) |
| `resume_set_b_left.cmd`, `resume_set_b_right.cmd`, `resume_set_c_left.cmd`, `resume_set_c_right.cmd` | Gloves on, no camera | the same with `--resume latest` (above) |
| `record_set_a.cmd`, `resume_set_a.cmd` | Camera only, bare left hand | Set A, the grasps (`docs\grasp_recording.md`, steps 7 and 8b) |
| `rehearse_transitions.cmd` | No hardware (mock camera) | the transition recorder on a synthetic hand: `open_hand>p1_cylindrical,p1_cylindrical>p1_spherical`, 2 takes each, into `recordings\protocol_mock\transitions\` (`docs\grasp_recording.md`, step 8d) |
| `record_transitions_left.cmd` | Camera only, bare left hand | the transition recorder, 3 kept takes of each pair given after its name, into `recordings\protocol\transitions\`; with no pair it prints how to give them and records nothing (steps 8e and 8f) |
| `rebuild_professor_grasp_set.cmd` | No hardware | the professor's grasp folder rebuilt, with every transition session in `recordings\protocol\transitions\` (`docs\grasp_recording.md`, step 10) |
