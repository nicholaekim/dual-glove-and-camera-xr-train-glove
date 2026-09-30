# When the camera loses the hand

The Ultraleap camera sometimes stops tracking the hand for a moment. The
tracking service itself is healthy: its log shows 90 Hz, no dropped frames
and about 10 ms of latency, and earlier bare-hand static poses tracked 97 to
100 percent of their frames. So a loss happens because of something in that
moment: where the hand is, how it is turned, what shape it makes, how fast it
moves, the light in the room, or the lenses. This page is the checklist that
removes most of those causes, then a 60 second test that records every loss
and says which cause it was.

## The physical checklist

Go through this before every session, top to bottom.

1. The module lies flat on the desk with the lenses facing straight up.
2. Wipe the two lenses with a soft cloth.
3. The camera cable goes straight into the laptop. No hub, no extension.
4. The laptop is on its charger.
5. No sunlight on the module, and no other infrared source near it: no
   halogen or incandescent lamp, no heater, no other depth camera.
6. Nothing shiny under the hand: no phone, watch face, foil or glossy paper
   on the desk in the camera's view.
7. Sleeves pushed up above the wrist. Take off a watch or bracelet.
8. Only one hand over the module. The other hand rests on the desk away
   from it.
9. The hand is 25 to 35 cm above the module, over the middle of it.
10. The palm faces the lenses roughly. Tilt it less than about 45 degrees.
11. Move slowly, especially when closing the hand.

## The 60 second test

### Step 1. No hardware: open PowerShell in the repo folder

```
cd "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
```

### Step 2. Camera only, bare left hand: run the test

```
.venv\Scripts\python.exe scripts\leap\tracking_quality.py --hand left
```

It prints the plan, opens the camera window, gives you 3 seconds to put the
hand over the module, then beeps and runs for 60 seconds. A beep marks each
new instruction, and the instruction is also the caption of the camera
window, so keep your eyes on the window. Do each one slowly:

| From | What to do |
|------|------------|
| 0 s  | hold the hand open, palm toward the lenses, 30 cm above the middle of the module, and keep it still |
| 10 s | move the hand slowly up to 50 cm, then slowly back down to 30 cm |
| 20 s | at 30 cm, move slowly out to the left and back, then out to the right and back |
| 30 s | turn the palm slowly away from the lenses, then back |
| 38 s | close the hand slowly into a fist, then open it |
| 43 s | pinch the thumb and index slowly, then open |
| 48 s | hold each of the paper grasps you like, one after another, slowly |

Some of these are meant to lose the hand (50 cm, the far side, the palm
turned away): the test finds out where your camera's limits are, so the
report can tell a loss you caused on purpose from one you did not.

At the end it writes two files into `results\diagnostics\` and prints the
report:

    tracking_quality_<stamp>.csv   one row per tracking frame: the hand's
                                   position, height, angle, grab, the image
                                   brightness and the device status
    tracking_quality_<stamp>.txt   the report, the same text as printed

The last line starts with `VERDICT:`. Ctrl+C stops early and still writes
both files.

## Reading the report

- **Tracked**: the share of frames in which the hand was tracked, over the
  whole run and from the moment the hand was first seen.
- **Tracker's label for the hand**: the tracker's own left or right guess.
  It is often wrong for a left hand; the test follows the hand, not the
  label.
- **Image brightness**: how bright the infrared picture is with no hand in
  view. An empty desk measured 5.8 of 255. Above 50, or more than 1 percent
  of the picture saturated, is a bright background.
- **Device status**: the camera's own warning flags (smudged lenses,
  infrared interference, low resource). "no warning flags" is normal.
- **Losses**: every time the hand was gone for more than 100 ms, or came
  back under a new hand id (the tracker lost it and found it again). For
  each one: when, how long, where the hand was just before (height, how far
  off the module's axis, how far the palm was turned from the lens, grab,
  speed) and why.
- **Causes, most losses first**: a loss can have more than one cause, and
  every cause of every loss is counted, so the first row is the one fix that
  would have prevented the most losses.
- **Fix to try first** and **VERDICT**: that fix. Apply it, then run the
  test again and compare the number of losses.

The report says so when something could not be measured: no infrared image
arrived (turn on "Allow Images" in the Ultraleap Control Panel), or the
service sent no device status. It never guesses.

## What each cause means and its fix

| Cause | What was measured | Fix |
|-------|-------------------|-----|
| too high | the palm more than 45 cm above the module | keep the palm 25 to 35 cm above the module |
| too low | the palm less than 12 cm above the module | raise the palm to 25 to 35 cm above the module |
| off centre | the palm more than 60 degrees off the module's axis (the field is 170 degrees wide, but its edge does not track well) | keep the hand over the middle of the module, not out at the side |
| palm turned away | the palm more than 60 degrees from facing the lenses | turn the palm back toward the lenses; tilt it less than about 45 degrees |
| closed hand from below | grab above 0.8 with the palm turned more than 40 degrees from the lenses: the curled fingers hide each other | open the hand over the module first, then close it slowly with the palm toward the lenses |
| moving fast | the palm moving faster than 0.5 m/s just before | move slowly, well under half a metre per second |
| frame rate low | the service's frame rate under 80 Hz in the 2 seconds before | plug the camera straight into the laptop (no hub), close other programs that use the camera, keep the laptop on its charger |
| bright background | with no hand in view, the infrared picture brighter than 50 of 255 or more than 1 percent saturated | keep sunlight and other infrared sources off the module, and nothing shiny under the hand |
| device status: lenses smudged | the camera reported smudged lenses | wipe the two lenses with a soft cloth |
| device status: infrared interference, robust mode | the camera switched to robust mode because of infrared light | take the module out of sunlight and away from other infrared sources |
| device status: low resource mode | the camera is short of USB bandwidth or processing | plug the camera straight into the laptop (no hub) and close other programs that use the camera |
| device status: a failure code | the camera reported a failure (USB, firmware, calibration) | unplug the camera, wait 5 seconds, plug it straight into the laptop, then run `scripts\leap\check_setup.py` |
| unexplained | none of the above | wipe the lenses, check for sunlight or another infrared source, and watch the camera window at the moment it loses the hand |

The device status is read from the raw value the service sends. The Python
bindings' own decoding of it reports "bad calibration" and "bad transport"
whenever the camera is simply streaming, so it is not used. The service has
no "low frame rate" or "low USB bandwidth" flag; the frame rate is measured
from the tracking frames instead.

## The same evidence during a session

- The camera window has a line near the bottom:
  `tracking: 90 Hz, lost 2 times this minute, last: too high`. It is the same
  classifier on the window's own stream, so it names the cause of a loss
  while you are still holding the pose.
- When the grasp recorder rejects a take for tracking, the reason names the
  losses: how many, and for the longest, where the hand was and why, for
  example `tracked 72 percent: lost 3 times, longest 1.4 s with the hand at
  49 cm (too high: keep the palm 25 to 35 cm above the module); the gate
  needs 90 percent`. Every loss of every take, kept or not, is in the take's
  `meta\<take>.json` under `gate.losses`. A recording has no infrared picture
  and no device status in it, so for a take only the hand's own causes are
  judged.

## Without the camera

### Step 1. No hardware: rehearse the test on a scripted hand

```
.venv\Scripts\python.exe scripts\leap\tracking_quality.py --mock --seconds 60 --no-view
```

The mock hand is lost on purpose in every way the report can explain, so
this shows what a report with every cause looks like. Nothing in it is
evidence about the real camera, and the report says so.

### Step 2. No hardware: print the report of an earlier run again

```
.venv\Scripts\python.exe scripts\leap\tracking_quality.py --from results\diagnostics\tracking_quality_20260930_164433.csv
```

Put the name of your own CSV in place of the example. It also reads a
recorded take's `leap\<take>.jsonl`.
