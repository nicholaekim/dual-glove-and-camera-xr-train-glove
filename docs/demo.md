# Demo runbook: the professor's visit

One window shows three hands per side, moving: the **glove** hand (orange),
the **camera** hand (blue, Ultraleap) and the **fused** hand (green). Under
the fused hand, badges say where each finger's curl and spread came from in
this frame, where the thumb came from and which fingers the rail override is
holding. Under the glove and camera hands is a pose guess.

## Before he arrives

1. Ultraleap Stereo IR 170 on USB, flat on the desk, lens up. The Ultraleap
   Tracking service must be running.
2. Both StretchSense gloves charged and on. XR Trainer running and
   streaming OSC to port 9002, both hands.
3. Run the replay once (below). It takes about 35 s to read and fuse
   sync_day2 before the window opens, so start it before he walks in.
4. Optional check that the live path still matches the offline report
   (about 50 s): `.venv\Scripts\python.exe scripts\fuse_live.py --replay
   recordings\sync_day2` must end with "Live path agrees with fuse_all on
   every take."

## The two commands

```powershell
.venv\Scripts\python.exe scripts\demo.py --replay recordings\sync_day2
.venv\Scripts\python.exe scripts\demo.py --live
```

**Replay** needs no hardware. It plays the recorded session at real speed,
take after take (take 1 of every pose, then take 2, and so on), and loops.
The header shows the take and its true pose. Keys: `space` pause, `n` next
take, `r` restart this take, `+` / `-` speed, `q` quit.

**Video:** add `--video clip.mp4` to the replay to write it to an MP4 at real speed instead of a window (every take once; `--takes 12` keeps take 1 of each pose, both hands, about 65 s).

**Live** runs the same warm-up as `fuse_live.py`, LEFT hand then RIGHT hand,
with the steps shown in the window, on the console and in the camera window:
hold the hand open 18 to 28 cm over the module until it is acquired, then
open palm flat to the camera (4 s), then a fist (4 s). Then every fused
frame is drawn as it is made. `q` quits. It first spends about 35 s learning
the pose centroids from sync_day2. `--hand right` shows one bigger row: on
the laptop screen the two-hand window is scaled down to fit.

**Where the data goes:** every live run saves into
`recordings\demo\<YYYY-MM-DD_HHMM>_<hands>\` (the window's footer names it):
the fused frames (`<hands>.jsonl`, both hands in one file with `--hand
both`), the warm-up and summary logs, the fitted `template_<hand>.json`,
`demo.mp4` (every frame the window drew, 30 fps), `snapshot.png`, and a
`README.txt` listing each file with the duration, paired share and camera use
per DOF. When you quit, the console's last line is `Data for this demo:
<folder>` and the folder opens itself in File Explorer. `--out PATH.jsonl`
puts it all beside PATH; `--no-save` saves nothing.

If the hardware misbehaves, fall back to the replay: it is the same fusion
code.

## What he will see

A row per hand. Each hand is drawn looking at the palm, fingers up, with the
wrist rotation taken out, all three at the same fixed millimetres per pixel,
so the hands can be compared finger by finger. A stream not seen for half a
second goes grey and says "no hand". Under the fused hand:

```
thumb    direction: camera
index    curl: glove     spread: camera
...
override: none
camera fresh   glove 60 Hz
```

`glove` is orange and `camera` is blue. Under the other two hands: the pose
guess (nearest centroid on the fused hand, the report's own features and
classifier), the margin to the runner-up, and in a replay the true pose with
a tick or a cross. In a replay the take on screen is left out of the
centroids.

## What to point at

- **Spread comes from the camera.** Open palm or peace: the spread words turn
  blue. The glove has no sensor for spread; compare the glove's fan of
  fingers with the camera's.
- **Pinch and the thumb come from the camera.** `thumb direction: camera` on
  pinch and thumbs_up. Thumb opposition is what made pinch look like an open
  palm to the glove alone.
- **Bend comes from the glove.** On a fist, every curl stays orange. The
  camera sees less of a closed gloved hand (fingers hide behind the palm);
  the glove measures each bend directly, occluded or not.
- **The override on a pinch.** In a pinch the glove's index sits pinned at its
  straight reading while the camera sees it bent. After a few frames the
  badge turns to `index curl: camera (override)` and `override: index`. The
  left-hand pinch takes show it for about 90 % of their frames.
- **The pose guess.** Tick or cross against the true pose, and the margin
  (small margin, unsure guess).

## Known limits: say these out loud

- **Six static poses** (fist, index_point, open_palm, peace, pinch,
  thumbs_up), each held for a few seconds. No transitions, no dynamic
  gestures; the guess only knows these six.
- **One operator, one glove pair.** Everything learned (the profile, the lags,
  the bone fit, the centroids) is this person on these gloves.
- **The right glove is session-dependent.** In sync_day1 and sync_day2 its
  middle, ring and pinky read partly straight while the camera reads them
  curled (`profiles/default.json` keeps them out of the thumb vote). In the
  replay the demo's guess is right on 86 % of frames overall and on at least
  91 % of every left-hand pose, but right-hand index_point (34 %) and peace
  (56 %) are mostly read as open_palm. The report's leave-one-take-out score
  on the same session: glove only 39/60, camera only 59/60, fused 52/60. On
  held poses the camera alone does better; the fusion's case is the finger
  the camera cannot see.
- **Live is a short calibration.** The warm-up is 8 s per hand, not a whole
  session. The right glove trails the camera by about 0.47 s on this laptop,
  so the right fused hand lags a movement by about half a second.
