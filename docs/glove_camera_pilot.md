# Your Reality Gloves and Ultraleap: what is ready to record

Checked 2026-10-25. The operator chose to prepare the procedure and record
later. No new physical capture was made today. Evidence and the saved-data
comparison are in `results/glove_camera_pilot_20261025/`.

## What equipment we can identify

Your XR Train logs identify **StretchSense Reality Glove (L)** and
**Reality Glove (R)**, family `REALITY`. The log is
`C:/Users/nkim2/AppData/LocalLow/StretchSense/XR Train/Player.log`, last
written 2026-10-23. The September evidence folder also names Reality Gloves.
This establishes the product family. Hardware revision, physical sensor
count and the anatomical location of each raw sensor remain unconfirmed.

XR Train sends `/v1/animation/kinematic/all` to `127.0.0.1:9002`.
The supported packet has five header fields and 26 joints, each with a
position and quaternion. Its 187 arguments are **not 187 physical sensors**.
The existing recorder replaces string labels with numeric placeholders,
so historical skeleton files do not preserve the device identity fields.
The new bounded `probe` command preserves those labels and inventories
the addresses that actually arrive, including a raw capacitance address
if XR Train sends one. No capacitance stream arrived during today's probe.
That probe received no packets of any kind.

The saved XR Train log has `Pinch=False`, `FingerSplay=False` and
`StudioGloveMode=False`. These are software feature flags for that session.
They are not a physical sensor inventory.

The manufacturer's [Reality XR Train setup guide](https://stretchsense.atlassian.net/wiki/spaces/rxt/pages/760315905/StretchSense+Reality+XR+Train+Glove+Setup),
read 2026-10-25, states that Basic articulation does not provide the full
thumb range or finger splay. It also distinguishes articulation training
from controller gesture training. This supports a software-output limit;
it does not establish that the glove hardware has no sensors for those
motions. Earlier descriptions of the glove as categorically unable to
sense spread should be read with this qualification.

## What the saved recordings actually show

The fresh audit reads the left-hand take 1 of open palm, fist and pinch in
`recordings/sync_day2/`. These are older real simultaneous recordings,
not today's experiment. It hashes all six recordings and their three sidecars and reports all
26 joints' three rotation components in `glove_angle_ranges.csv`.

- The glove's index MCP flexion spans 0.00 to 89.76 degrees and its PIP
  spans 4.13 to 90.00 degrees across the three takes.
- The four finger metacarpals have zero flexion variation. Their splay
  values remain 5, 0, -10 and -15 degrees. These values cannot fill the
  missing finger CMC measurements.
- The thumb metacarpal's decomposed coordinates vary. That does not show
  that each component is independently sensed: a coupled rotation can
  change multiple components. Its anatomical TMC mapping remains unverified.
- No raw capacitance-to-anatomy channel map is present in these files.

The existing pilot comparison measures MCP and PIP flexion for the four
fingers, eight joint coordinates. It uses the last three seconds of each
hold and nearest timestamps within 50 ms. The median of the eight joint
median absolute disagreements was 8.11 degrees for open palm, 14.77 for
fist and 9.63 for pinch. The respective camera-frame pair counts were
269, 269 and 271. The same glove sample can pair with more than one camera
frame; these are not independent participant repetitions.

Those are disagreements between two solved skeletons. The clock pairs
camera acquisition against glove OSC receipt. Receipt time includes
unknown glove processing and transport latency. No latency correction was
fitted, and these numbers do not establish anatomical accuracy.

## When you are ready

1. Connect the Ultraleap directly by USB. Confirm it appears in its Control
   Panel. Place it lenses up on the table, away from direct sunlight.
2. Turn on the Reality Gloves and open XR Train. Connect them to their
   correct hands. Wear and calibrate the left glove using the manufacturer's
   articulation instructions. Save the profile and record its identity for
   any research dataset. Keep the right hand out of the camera view.
3. Enable the OSC kinematic stream to `127.0.0.1:9002`. Close other Python
   OSC viewers or recorders so only this recorder owns port 9002.
4. From the repository root, run:

   ```powershell
   & .\launchers\glove_camera_pilot.ps1
   ```

5. The launcher checks the camera and listens for OSC labels. It stops with
   a reason if the device or left-hand skeleton is missing. Press Enter
   at its prompt when you are ready to record.
6. Follow its open-palm, fist and pinch prompts. Before every pose, show an
   open left hand, palm toward the lenses, 18-28 cm above the camera. Let it
   track the hand before changing pose. Each hold records for five seconds.
   This first pilot is static; slow bending sweeps and latency testing come
   after confirming usable pairing and tracking.
7. The launcher writes a new timestamped folder under
   `recordings/glove_camera_pilot/`. It records both streams in `capture/`,
   the actual OSC identity check in `osc_probe/`, and the comparison in
   `comparison/`. Rejected attempts remain available for review. If a pose
   is missing, the audit reports the failure instead of claiming success.

The launcher uses the existing `record_simultaneous.py` and its normal
tracking and pose checks. It does not alter either device's settings,
the original dataset, or old results. A folder with an existing comparison
is refused rather than overwritten.

## Commands already exercised

Run a fresh offline audit with a new output directory:

```powershell
.venv\Scripts\python.exe scripts\glove_camera_pilot_check.py audit --session recordings\sync_day2 --hand left --out results\my_new_glove_camera_audit
```

Inspect the live OSC stream for five seconds without recording a take:

```powershell
.venv\Scripts\python.exe scripts\glove_camera_pilot_check.py probe --seconds 5 --out results\my_new_osc_probe
```

Rehearse the paired recording without devices:

```powershell
.venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --mock-glove --mock-leap --poses open_palm --takes 1 --duration 4 --prep 0 --settle 0.2 --acquire-timeout 8 --retries 0 --no-view --out-dir results\my_new_mock_pilot
```

Read that mock capture as synthetic evidence:

```powershell
.venv\Scripts\python.exe scripts\glove_camera_pilot_check.py audit --session results\my_new_mock_pilot --hand left --poses open_palm --data-origin synthetic --out results\my_new_mock_comparison
```

Still outstanding: the new live pilot, the hardware revision and raw
sensor-channel documentation from StretchSense, and an independent
anatomical reference for the six missing coordinates. Neither device is
automatically an anatomical reference for the other.
