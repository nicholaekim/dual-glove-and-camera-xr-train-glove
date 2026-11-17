# Plan: one take per pose, checked by a vision-language model

Written 11 Nov 2026. Nothing below is built yet. Status of each fact: measured
on this machine today unless marked "estimate".

## Context

Request: for both recordings (camera only with the Ultraleap, and gloves on
with the Ultraleap watching), record **one take per pose** instead of three,
and let a **vision-language model (VLM) verify** that the take shows the pose
that was asked for, "something like" arXiv 2512.21970.

- **The paper link does not match.** arXiv 2512.21970 is "StereoVLA: Enhancing
  Vision-Language-Action Models with Stereo Vision" (Deng et al.), a robot
  control paper. It has no verifier step. This plan therefore follows the
  general VLM-as-verifier pattern (show the model the picture and a checklist,
  get a structured verdict, redo on a fail). If a different paper was meant,
  its prompt and inputs replace section 3.
- **Why a VLM and not the tracker's own angles.** The recordings exist to test
  the sensors against the constraint ranges. Checking a pose with the same
  tracker that measured it is circular. A model that looks at the raw infrared
  picture is independent of the tracker.
- **What already exists** (survey of both recorders):
  - Both save one hand-cropped infrared still per take, taken at the middle of
    the hold: `stills/<take>.png` (camera recorder) and `stills/<stem>.jpg`
    (glove recorder). That still is the verifier's input.
  - `scripts/leap/record_poses.py` protocol mode: tracking gate
    (`summarise_take`), then a human review (`Reviewer.decide`), retries with
    `--retries`.
  - `scripts/record_simultaneous.py`: tracking checks, then a numeric shape
    check (`_check_shape` -> `leap_hand.pose_check.check_pose`), reject and
    retry through `_reject`. No human review step.
  - Every pose in both new lists carries per-digit states (`index_state` ...
    `thumb_state`, `spread_state`, `wrist_state`) in
    `protocols/pose_catalogue_100.csv` and `protocols/glove_planar_poses_50.csv`,
    and has two MyoHand renders (`<id>_palm.png`, `<id>_side.png`) under
    `results/pose_catalogue/renders/` and `results/glove_planar_poses/renders/`.
- **Which model can run here.**
  - Local: Ollama is installed with one vision model, `qwen3.8` (27B, 17.7 GB).
    The laptop has 15.7 GB RAM and a 6 GB GPU. One judgement of one still did
    **not finish in 9 minutes**. Not usable.
  - Cloud: no API key or login is set up (`ANTHROPIC_API_KEY` and the others
    unset, `ant` CLI not installed).

## 1. Decisions that need the user

1. **Backend.** Recommended: Claude API through the `anthropic` Python SDK.
   It needs a credential the user sets up (an `ANTHROPIC_API_KEY` environment
   variable); the infrared hand crops leave this machine. Alternative: pull a
   small local vision model with Ollama (a download of a few GB that fits the
   6 GB GPU); private and free, accuracy unknown until section 5 measures it.
2. **Spend for the accuracy test** in section 5 (about 370 calls). Estimate:
   under 15 USD on `claude-opus-5-5`, under 0.50 USD on `claude-haiku-5-5`.
3. **The right paper**, if 2512.21970 was a typo.

## 2. One take

- `protocols/glove_planar_poses_50.json`: `takes_per_item` 3 -> 1, via its
  generator `scripts/leap/glove_planar_poses.py` (also `RECORD_COMMAND`
  `--takes 3` -> `--takes 1` and the "three takes per pose" sentence in
  `docs/glove_planar_poses_50.md`). `tests/test_glove_planar_poses.py:137`
  asserts 3 and changes with it.
- The 100-pose catalogue gets a protocol file with `takes_per_item` 1
  (section 4).
- **Left alone:** the older sets (`protocols/grasps.json`, the professor's
  Sets A to C, the ten paired poses) and the recorders' own defaults. Those
  hand-ins were defined with their take counts.

## 3. The verifier

New module `src/leap_hand/vlm_verify.py`, no window, no hardware:

- `checklist(pose_row) -> dict`: the expected state of each digit, the spread
  and the thumb, read from the pose's CSV row.
- `verify(still, references, pose_row, model) -> Verdict`. One request with
  three images (the take's infrared still, the palm render, the side render)
  and the checklist as text. The model answers what it **sees** first, then
  whether it matches:

      {"observed": {"thumb": ..., "index": ..., "middle": ..., "ring": ...,
                    "little": ..., "spread": ...},
       "match": "yes" | "no" | "unsure", "confidence": 0.0-1.0,
       "reason": "one sentence"}

  Structured output (`output_config.format`), parsed with `json.loads`.
- **The verdict is computed in code**, from the observed states against the
  expected ones plus the model's own `match`: `pass`, `fail`, or `unsure`.
  Any error, timeout, refusal or unparsable reply is `unchecked`. `unsure`
  and `unchecked` are never treated as a pass.
- `MockVerifier(script)` returns scripted verdicts so every recorder test
  runs with no network.
- Model from `--vlm-model`, default `claude-opus-5-5`; section 5 also measures
  `claude-haiku-5-5` so the choice rests on measured accuracy and cost.
  Before writing the call, read `python/claude-api/tool-use.md` in the
  `claude-api` skill for the structured-output and refusal-fallback shapes.

## 4. Recorder changes

**Camera only (`scripts/leap/record_poses.py`, protocol mode)**

- Generator `scripts/leap/pose_catalogue.py` also writes
  `protocols/pose_catalogue_100.json` (items with `id`, `label`, `shape`,
  `images_dir`, `image` = `<id>_palm.png`, `takes_per_item` 1). This is also
  why the target pictures do not show today: the recorder reads JSON
  protocols only and the catalogue has none.
- New flag `--verify vlm|off`. In `attempt()`, after the tracking gate passes
  and before the human review:
  - `pass`: kept, `decided_by: "vlm"`, no keypress.
  - `fail`: concluded as rejected with the model's reason, retried up to
    `--retries`; the reason is shown beside the still and the reference.
  - `unsure` or `unchecked`: the existing human review decides, with the
    model's answer shown.

**Gloves on (`scripts/record_simultaneous.py`)**

- It reads only `--poses` ids, `POSE_HINTS` (7 entries) and
  `reference_poses/<pose>.png`. Add `--protocol <json>` so the 50 poses get
  their label, hint text and render from the protocol file.
- In `_attempt()`, next to `_check_shape`: `fail` rejects through `_reject`
  with the model's reason (existing retry loop); `unsure` or `unchecked`
  keeps the take and marks it, and the session ends with a list of takes
  for the operator to look at. This recorder has no human review step.
- `--mock-leap` writes no still today; give it the synthetic still the camera
  recorder's mock uses (`write_mock_still`).

**Both:** every take's meta gains a `vlm` block: model, prompt version,
verdict, expected, observed, reason, confidence, latency, and the still's
SHA-256. `session.json` take entries gain `vlm_verdict`. Rejected attempts
keep their still and the model's reply in `rejected/`.

## 5. Prove it before trusting it

`scripts/leap/vlm_verify_eval.py` runs the verifier over stills that already
exist and whose pose is known from the file name:

| Pose | Stills |
|---|---|
| pinch | 33 |
| thumbs_up, peace, open_palm, index_point | 31 each |
| fist | 30 |

165 of the recordings' stills are glove-on (`recordings/sync*`). Each still is
judged twice: against its own pose (should pass) and against one other pose
(should fail). Reported separately for bare hand and glove on: false accepts,
false rejects, unsure rate, seconds per call, cost.

**Rule fixed before the run:** the verifier decides on its own only for an
image kind where false accepts are at most 5 % on at least 100 mismatched
pairs. Otherwise it is advisory: in the camera recorder its answer is shown
in the human review; in the glove recorder it only flags. The six test poses
are simpler than many catalogue poses, so this is an upper bound on what to
expect, and the note in the hand-in says so.

## 6. Files

| File | Change |
|---|---|
| `src/leap_hand/vlm_verify.py` | new: checklist, request, verdict rule, mock |
| `scripts/leap/vlm_verify_eval.py` | new: accuracy and cost on existing stills |
| `scripts/leap/record_poses.py` | `--verify`, hook in `attempt()`, meta block |
| `scripts/record_simultaneous.py` | `--protocol`, hook in `_attempt()`, mock still, meta block |
| `scripts/leap/pose_catalogue.py` | writes `protocols/pose_catalogue_100.json`, record command in the doc |
| `scripts/leap/glove_planar_poses.py` | one take, record command |
| `pyproject.toml` | optional extra `vlm = ["anthropic"]` |
| `tests/test_vlm_verify.py` and the two recorder test files | new and extended |
| `docs/pose_catalogue_100.md`, `docs/glove_planar_poses_50.md`, `README.md` | commands and the verifier's limits |
| `GATES_vlm_verifier.md` | acceptance gates |

## 7. Order of work

Every step is **no hardware** until the last.

1. Gates file; tests for the checklist and the verdict rule (written first).
2. `vlm_verify.py` with the mock; tests pass.
3. Catalogue JSON protocol; glove list to one take; docs regenerate.
4. Camera recorder hook; mock runs: pass keeps one take with
   `decided_by: "vlm"`; fail then pass keeps attempt 2 and files attempt 1
   under `rejected/` with the reason; unsure asks the human; an API error is
   `unchecked` and asks the human.
5. Glove recorder: `--protocol`, hook, mock still; the same four cases.
6. The real request (needs the credential), then the section 5 run and its
   report (needs the spend approved).
7. **Camera only, bare hand:** record three catalogue poses live, one take
   each, once with the right pose and once deliberately wrong.

## 8. Gates (to go in `GATES_vlm_verifier.md`)

- Verdict rule: every combination of expected, observed and `match` gives the
  outcome written in the test table; an unparsable reply is `unchecked`.
- Each recorder, mock run: one accepted take per pose; the four cases of
  step 4; meta and `session.json` carry the `vlm` block.
- Both docs' record commands say `--takes 1` and run on the mock.
- The evaluation report exists with its counts, and the recorder's default
  (decides or advisory) is the one the 5 % rule gives.
- Whole suite passes (run with a one-hour cap; it takes about 12 minutes).
- Nothing committed.

## 9. What this does not do

- It does not check joint angles. That stays with the tracking gate and the
  later range comparison.
- It judges one still from the middle of the hold, not the whole take.
- With gloves on, the picture shows the glove, not skin. Accuracy there is
  unknown until section 5 measures it.
