@echo off
rem Redo of three poses after the independent inspection of the hand-in of 2026-11-05:
rem crossed_fingers, thumb_opposition and pinch, one take each, then the tool and the visual-check sheet.
rem --first-take auto makes these take 3 of each pose; nothing is overwritten.
cd /d "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
set POSES=crossed_fingers,thumb_opposition,pinch
echo Redo of three poses. GLOVES ON + CAMERA, LEFT hand.
echo Order: crossed fingers, thumb opposition, pinch. One take of each; they become take 3 of each pose,
echo nothing is overwritten.
echo.
echo BEFORE RECORDING:
echo  - Calibrate ONCE in XR Trainer at the start. This is a new session, so a new calibration is
echo    allowed; it is written into every take's --note. Do not recalibrate again during this run.
echo  - Right hand resting on the table, outside the camera's view.
echo  - Camera flat, lenses up. Left hand 20 to 28 cm above the module.
echo  - Every take starts from an OPEN PALM, then the pose the window names.
echo  - The window shows the professor's picture of each pose in its top-right corner: copy it.
echo  - The photo must SHOW the pose:
echo    crossed fingers  = middle finger crossed over the index, the crossing turned toward the camera,
echo                       other fingers curled.
echo    thumb opposition = thumb tip touching the pinky tip (or the base of the pinky), the other
echo                       fingers extended, palm toward the camera.
echo    pinch            = thumb tip touching the index tip, the contact turned toward the camera,
echo                       other fingers extended.
echo.
pause
echo Recording the three poses:
.venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --poses %POSES% --takes 1 --first-take auto --duration 5 --out-dir recordings\ten_poses_left --note "redo of three poses after the independent inspection of the hand-in of 2026-11-05: new calibration; crossed fingers visibly crossed; thumb tip to pinky tip; pinch contact facing the camera"
if errorlevel 1 goto failed
echo.
echo Running the paired-pose tool on the whole session:
.venv\Scripts\python.exe scripts\paired_pose_compare.py recordings\ten_poses_left
echo.
type results\paired_pose\ten_poses_left\comparison.csv
echo.
echo NEXT, no hardware: open results\paired_pose\ten_poses_left\visual_checks.csv. It has three new blank
echo rows, one per new take: look at each take's figure.png and write yes or no in intended_pose_seen
echo (did the photo and both models show the intended pose?). Then run this one command:
echo   .venv\Scripts\python.exe scripts\paired_pose_compare.py --select results\paired_pose\ten_poses_left
pause
exit /b 0

:failed
echo The recorder stopped and reported a failure. Nothing was deleted. Fix the setup and run this launcher
echo again: every pose continues at its next take number, nothing is overwritten.
pause
exit /b 1
