@echo off
rem Ten-pose session, part 1 of 2: the first-take gate (open_palm, one take).
rem Optional argument: a note added to this run's --note, e.g. the reason for an
rem unavoidable recalibration:  record_ten_poses_1_first_take.cmd "recalibrated before this run: glove slipped"
cd /d "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
set EXTRA=%~1
echo Ten-pose session, part 1 of 2: OPEN PALM, one take. GLOVES ON + CAMERA, LEFT hand.
echo.
echo BEFORE RECORDING:
echo  - ONE calibration in XR Trainer at the start of the session. No recalibration between parts.
echo    If one is unavoidable: run this launcher with the reason in quotes after its name (it goes
echo    into the take's --note), and write it in results\paired_pose\ten_poses_left\README.md too.
echo  - Right hand resting on the table, outside the camera's view.
echo  - Camera flat, lenses up. Left hand 20 to 28 cm above the module.
echo  - Every take starts from an OPEN PALM.
echo.
pause
.venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --poses open_palm --takes 1 --first-take auto --duration 5 --out-dir recordings\ten_poses_left --note "ten-pose paired session for the professor, left hand, part 1 (first-take gate) %EXTRA%"
if errorlevel 1 (
  echo The recorder stopped and reported a failure. Nothing was deleted. Fix the setup and run part 1 again.
  pause
  exit /b 1
)
echo.
echo Running the paired-pose tool on the session:
.venv\Scripts\python.exe scripts\paired_pose_compare.py recordings\ten_poses_left
echo.
type results\paired_pose\ten_poses_left\comparison.csv
echo.
echo FIVE CHECKS on the NEWEST open_palm row above (the highest take number):
echo   1. moment_source = "still capture time"
echo   2. cam_dt_ms and glove_dt_ms both within -50..50
echo   3. stillness = "still"
echo   4. handedness = "same"
echo   5. open results\paired_pose\ten_poses_left\open_palm_left_take^<N^>_*\figure.png and confirm by eye
echo      that the photo and both models show an open left palm.
echo.
:ask
set ANSWER=
set /p ANSWER=Did all five checks pass? (y/n)
if /i "%ANSWER%"=="y" goto passed
if /i not "%ANSWER%"=="n" goto ask
:reason
set REASON=
set /p REASON=Reason:
if not defined REASON goto reason
.venv\Scripts\python.exe scripts\paired_pose_compare.py --gate-fail results\paired_pose\ten_poses_left --newest-of open_palm --reason "%REASON%"
if errorlevel 1 (
  echo The setup-gate failure could not be recorded. Write the take and the reason down by hand.
  pause
  exit /b 1
)
echo.
echo Fix the setup and RUN PART 1 AGAIN. Nothing is deleted: the failed take stays, the next run records
echo the next take number, and the take selection rejects the failed take with the reason above.
pause
exit /b 1

:passed
echo All five checks passed. Next: run record_ten_poses_2_rest.cmd.
pause
exit /b 0
