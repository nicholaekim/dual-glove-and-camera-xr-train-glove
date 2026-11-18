@echo off
cd /d "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
echo Still-timestamp dry run. CAMERA ONLY, bare LEFT hand, no gloves (the glove stream is simulated).
echo SCOPE: this validates the camera's timestamp-sidecar path only, not two-sensor timing, calibration or tracking quality.
echo Camera flat, lenses up. When the window asks, hold an OPEN PALM 20 to 28 cm above the module. One 5 s take.
echo.
.venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --mock-glove --no-pose-check --poses open_palm --takes 1 --duration 5 --out-dir recordings\sidecar_dry_run --note "still timestamp dry run: mock glove, bare left hand"
echo.
echo Checking the still's timestamp through the paired-pose tool:
.venv\Scripts\python.exe scripts\paired_pose_compare.py recordings\sidecar_dry_run
echo.
type results\paired_pose\sidecar_dry_run\comparison.csv
echo.
echo PASS if the moment_source column above says "still capture time". FAIL if it says "still save time" or "still file write time".
pause
