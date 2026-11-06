@echo off
rem Ten-pose session, part 2 of 2: the nine other poses twice, interleaved, then
rem open_palm's second take, then the tool and the visual-check sheet.
rem Optional argument: a note added to every --note of this run (e.g. an unavoidable recalibration).
cd /d "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
set EXTRA=%~1
set POSES=finger_spread,fist,thumb_opposition,index_point,crossed_fingers,thumbs_up,bent_wrist,peace,pinch
echo Ten-pose session, part 2 of 2. GLOVES ON + CAMERA, LEFT hand.
echo Order, the same in both rounds: finger spread, fist, thumb opposition, index point, crossed fingers,
echo thumbs up, bent wrist, peace, pinch. Round 1 = take 1 of each, round 2 = take 2 of each,
echo then open palm's second take.
echo.
echo BEFORE RECORDING:
echo  - Same calibration as part 1: do NOT recalibrate in XR Trainer between parts.
echo    If one is unavoidable: run this launcher with the reason in quotes after its name (it goes
echo    into every take's --note), and write it in results\paired_pose\ten_poses_left\README.md too.
echo  - Right hand resting on the table, outside the camera's view.
echo  - Camera flat, lenses up. Left hand 20 to 28 cm above the module.
echo  - Every take starts from an OPEN PALM, then the pose the window names.
echo.
pause
echo Round 1 of 2:
.venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --poses %POSES% --takes 1 --first-take auto --duration 5 --out-dir recordings\ten_poses_left --note "ten-pose paired session for the professor, left hand, part 2 round 1 %EXTRA%"
if errorlevel 1 goto failed
echo Round 2 of 2:
.venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --poses %POSES% --takes 1 --first-take auto --duration 5 --out-dir recordings\ten_poses_left --note "ten-pose paired session for the professor, left hand, part 2 round 2 %EXTRA%"
if errorlevel 1 goto failed
echo Open palm, second take:
.venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap --hand left --poses open_palm --takes 1 --first-take auto --duration 5 --out-dir recordings\ten_poses_left --note "ten-pose paired session for the professor, left hand, part 2 open palm second take %EXTRA%"
if errorlevel 1 goto failed
echo.
echo Running the paired-pose tool on the whole session:
.venv\Scripts\python.exe scripts\paired_pose_compare.py recordings\ten_poses_left
echo.
type results\paired_pose\ten_poses_left\comparison.csv
echo.
echo NEXT, no hardware: open results\paired_pose\ten_poses_left\visual_checks.csv and, for every take,
echo look at its figure.png and write yes or no in intended_pose_seen (did the photo and both models show
echo the intended pose?). Then run this one command:
echo   .venv\Scripts\python.exe scripts\paired_pose_compare.py --select results\paired_pose\ten_poses_left
pause
exit /b 0

:failed
echo The recorder stopped and reported a failure. Nothing was deleted. Fix the setup and run part 2 again:
echo every pose continues at its next take number, nothing is overwritten.
pause
exit /b 1
