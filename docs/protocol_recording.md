# Runbook: recording Set B (finger flexion) and Set C (sequences)

The professor asked for single-finger flexion (Set B) and seven sequential
movements (Set C), recorded with the gloves. `scripts\record_protocol.py`
records both, one hand per run, with the Ultraleap camera running beside the
gloves as the reference. It is hands-free once it starts: every movement is
called by a beep, shown in words on a small window and printed in the
console. The plan behind it is `docs\grasp_and_flexion_protocol_plan.md`; the
files it writes are specified in `docs\protocol_formats.md`.

Every command below is typed in PowerShell in the repo folder
(`C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer`).

## Part 1: rehearse with no hardware

A rehearsal uses a mock glove and a mock camera, so its folders go under
`recordings\protocol_mock\` and its `session.json` says `"mock": true`:
rehearsals never sit beside real sessions, and the packager refuses them.

Step 1. **No hardware.** Rehearse Set B with the mock glove and mock camera
(about 15 seconds):

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set finger_flexion --hand left --items index_fast --mock-glove --mock-leap --no-view --no-open --calibrated-at 2026-09-28T14:00:00 --time-scale 0.25
```

The console ends with `1 accepted, 0 rejected` and the folder it wrote,
under `recordings\protocol_mock\finger_flexion\`.

Step 2. **No hardware.** Rehearse Set C the same way (about 15 seconds):

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set sequences --hand left --items seq3_alternating --takes 1 --mock-glove --mock-leap --no-view --no-open --calibrated-at 2026-09-28T14:00:00 --time-scale 0.25
```

The console ends with `1 accepted, 0 rejected` and the folder it wrote,
under `recordings\protocol_mock\sequences\`.

Step 3. **No hardware.** Rehearse a rejected take. This mock glove does the
warm-up and then keeps the hand open, so the take must be rejected:

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set finger_flexion --hand left --items index_fast --mock-glove-ignore-cues --mock-leap --no-view --no-open --calibrated-at 2026-09-28T14:00:00 --time-scale 0.25 --retries 0
```

The console says `REJECTED: the index moved only 0.0... of its warm-up range
(needs 0.60)`, and the attempt's files are in its folder under
`recordings\protocol_mock\finger_flexion\`, inside `rejected\`, with the
reason in `rejected\<take>.reason.txt`.

Step 4. **No hardware.** Check the folder from step 1 with the checker. Use
the folder path the console printed at the end of step 1:

```powershell
.venv\Scripts\python.exe scripts\check_protocol.py recordings\protocol_mock\finger_flexion\<folder from step 1>
```

It ends with `Every accepted take passes.`

## Part 2: the left hand

Step 5. **Gloves on + camera.** Set up the desk: the Ultraleap flat on the
desk, lens up, on USB; both gloves charged and on; XR Trainer running and
streaming to port 9002. Close any other recorder that listens on port 9002.

Step 6. **Gloves on + camera.** Record Set B with the left hand (7 takes,
about 7 minutes of recording):

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set finger_flexion --hand left
```

What happens, in order:

1. The console asks you to calibrate both gloves in XR Trainer. Do it, then
   press Enter. The time you press Enter is saved as the calibration time.
2. The recorder waits for the left glove, then for the camera: hold the
   left hand open, palm toward the lens, 18 to 28 cm above the module. The
   right hand rests on the table for the whole run. A camera window shows
   the IR image, the skeleton and the height against the band, so you can
   see where your hand is and how far each finger bends (`--hide-camera`
   runs without it; it was the default until 2026-10-01).
3. Warm-up: a beep and "OPEN PALM" for 3 seconds, a higher beep and "FULL
   FIST" for 3 seconds. This measures each finger's own range for today.
4. The takes: thumb, index, middle, ring, little finger, index slow, index
   fast. Each one shows its name for 3 seconds, then calls every bend, hold,
   straighten and rest with a beep. Move one finger only and keep the others
   straight. Follow the beeps, not the screen.
5. After each take the console says ACCEPTED or REJECTED with the reason. A
   rejected take is recorded again straight away, up to two more times.
6. At the end the console prints a table of every take and the folder, and
   the folder opens in File Explorer.

Step 7. **Gloves on + camera.** Check Set B for the left hand. Use the
folder path printed at the end of step 6:

```powershell
.venv\Scripts\python.exe scripts\check_protocol.py recordings\protocol\finger_flexion\<folder from step 6>
```

Step 8. **Gloves on + camera.** Only when step 6 or step 7 named a failed
take: record those items again, with their ids from the table, separated by
commas (for example `ring,index_fast`):

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set finger_flexion --hand left --items <failed ids>
```

Step 9. **Gloves on + camera.** Record Set C with the left hand (21 takes:
the seven sequences in three rounds, each round in a shuffled order, about
10 minutes of recording):

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set sequences --hand left
```

The same start as step 6 (calibration, Enter, camera, warm-up). Then every
step of a sequence is one beep and its words: make the shape, then hold it
until the next beep. A high beep means a finger flexes, a low beep means
fingers only straighten.

Step 10. **Gloves on + camera.** Check Set C for the left hand. Use the
folder path printed at the end of step 9:

```powershell
.venv\Scripts\python.exe scripts\check_protocol.py recordings\protocol\sequences\<folder from step 9>
```

Step 11. **Gloves on + camera.** Only when step 9 or step 10 named a failed
take: record one more take of each of those sequences:

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set sequences --hand left --items <failed ids> --takes 1
```

## Part 3: the right hand

Step 12. **Gloves on + camera.** Record Set B with the right hand. The left
hand now rests on the table:

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set finger_flexion --hand right
```

Step 13. **Gloves on + camera.** Check Set B for the right hand:

```powershell
.venv\Scripts\python.exe scripts\check_protocol.py recordings\protocol\finger_flexion\<folder from step 12>
```

Step 14. **Gloves on + camera.** Only when step 12 or step 13 named a
failed take: record those items again:

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set finger_flexion --hand right --items <failed ids>
```

Step 15. **Gloves on + camera.** Record Set C with the right hand:

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set sequences --hand right
```

Step 16. **Gloves on + camera.** Check Set C for the right hand. This is the
second hand, so the bands come from this session's own open-hand and
full-fist steps (plan, section 4):

```powershell
.venv\Scripts\python.exe scripts\check_protocol.py recordings\protocol\sequences\<folder from step 15> --bands auto
```

Step 17. **Gloves on + camera.** Only when step 15 or step 16 named a
failed take: record one more take of each of those sequences:

```powershell
.venv\Scripts\python.exe scripts\record_protocol.py --set sequences --hand right --items <failed ids> --takes 1
```

Step 18. **No hardware.** Write down every folder path the console printed
in part 2 and part 3. The hand-in folder is built from them by
`scripts\package_professor_set.py` (its `--flexion` and `--sequences`
options take several folders each).

## What you see and hear

The window shows only the take name, the words of the current cue and a
countdown. It never shows the camera image or a skeleton, on purpose: the
operator follows the beep, and a live skeleton invites shaping the movement
until the two sensors agree.

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
| `q` | stop the session after this take; everything recorded so far is kept |

The keys work on the cue window and in the console. Ctrl+C stops at once;
the take in progress is moved to `rejected\` as interrupted.

## What makes a take accepted

- Set B: the cued finger has to move at least 0.60 of its own open-to-fist
  range from the warm-up. The other four fingers are measured and reported,
  never failed on. The cycles are counted from the beeps and from the
  finger's peaks; a mismatch is a note, not a rejection.
- Set C: in the last 1.5 seconds of every step, a finger the step says is
  flexed has to read above 0.60 of its range, and a finger the step says is
  straight must not read above 0.60. A straight finger between 0.30 and 0.60
  is reported as coupling (the ring pulls its neighbours along) and does not
  reject the take.
- The session is refused at the warm-up when the glove's index, middle, ring
  or little finger moves less than 0.30 between the open palm and the fist.

## When something stops the run

| The console says | Do this |
|------------------|---------|
| `Only 0 glove packets ...` | Start streaming in XR Trainer, then run the same command again. |
| `The glove is streaming right, not left` | Put the glove on the hand named in the command, then run it again. |
| `No live tracking: ...` | Plug the camera in and start the Ultraleap Tracking service, then run it again. |
| `REFUSED: no acquirable left hand in 60 s` | Hold the hand open, palm toward the lens, 18 to 28 cm above the module, then run it again. |
| `REFUSED: warm-up: the glove barely moved ...` | Calibrate both gloves in XR Trainer again, then run it again and make a full fist. |

## Where the files go

One folder per run:
`recordings\protocol\<finger_flexion or sequences>\<date>_<time>_<hand>\`

| File | What it is |
|------|------------|
| `session.json` | the run: protocol file and version, hand, calibration time, round order and seed, every take with its verdict (rewritten after every take) |
| `warmup.json` | each finger's open, fist and span for this run, glove and camera |
| `glove\<take>.jsonl` | the glove at its full rate, every line tagged with session, item and take |
| `leap\<take>.jsonl` | the camera beside it, both hands if it saw two |
| `events\<take>.events.jsonl` | every cue at the moment of its beep, and every decision |
| `stills\<take>.png` | one hand-cropped IR still per take (stays in the repo, never handed in) |
| `rejected\` | every attempt that was not accepted, same layout, with `<take>.reason.txt` |

`--time-scale` (used in part 1) shortens every duration for a rehearsal and
is saved in `session.json`; a real session always runs at the default of 1.
A run with a mock sensor writes the same layout under
`recordings\protocol_mock\` instead, marked `"mock": true`.
