@echo off
rem The single-plane glove poses of the recording set, three takes each, GLOVE ONLY
rem (no camera), with a screenshot of the XR Train window for every take.
rem Optional argument: the pose id to start at, to carry on after a stop:
rem   record_glove_poses.cmd pc01
cd /d "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
set START=
if not "%~1"=="" set START=--start-at %~1
echo The glove poses of the recording set, 3 takes each. GLOVES ON, NO CAMERA, LEFT hand.
echo.
echo BEFORE RECORDING:
echo  - XR Train open, gloves connected, ONE calibration at the start. Leave the XR Train
echo    window open and NOT minimised (it may sit behind other windows): a picture of it
echo    is saved for every take.
echo  - No camera is used. Only the gloves and XR Train are needed.
echo  - The pose pictures beside you:
echo      results\glove_planar_poses\glove_planar_poses_recording_set.pdf
echo  - For each take the console names the pose and describes it, counts down 5 seconds
echo    with beeps on 3, 2, 1, then records 5 seconds. Get into the pose during the
echo    countdown and hold it until the low beep.
echo  - Fingers side by side, wrist straight: each finger bends only in its own plane.
echo  - To stop: Ctrl+C. To carry on later from a pose: record_glove_poses.cmd ^<pose id^>
echo.
pause
.venv\Scripts\python.exe scripts\glove\record_poses.py --hand left --pose-list protocols\glove_planar_poses_recording_set.json %START% --takes 3 --first-take auto --duration 5 --prep 5 --hz 0 --out-dir recordings\glove_planar_left --app-screenshot "XR Train" --note "single-plane glove poses, recording set, left hand, glove only, XR Train screenshot per take"
if errorlevel 1 (
  echo.
  echo The recorder stopped and reported a failure. Nothing was deleted.
  echo Read the message above, fix the setup, and run this launcher again.
  pause
  exit /b 1
)
echo.
echo Recordings: recordings\glove_planar_left   ^(the takes, their meta.json, and xr_train\^)
echo The summary above counts how many takes have an XR Train screenshot.
pause
exit /b 0
