@echo off
rem Professor's demo, NO HARDWARE fallback: the same flow as prof_demo.cmd with a mock glove and a
rem mock camera. A mock take has no photo, so the paired-pose tool refuses it and the flow ends at
rem the recorder; then this opens the PRE-RECORDED figure of the pose from the ten-pose session
rem (the accepted take named in results\paired_pose\ten_poses_left\selection.csv).
rem Usage: prof_demo_mock.cmd POSE, one of pinch (the default), open_palm, fist, peace
rem Mock takes go to recordings\prof_demo_mock, beside the real demo folder, never in it.
setlocal
cd /d "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
set PY=.venv\Scripts\python.exe
set REC=recordings\prof_demo_mock
set SEL=results\paired_pose\ten_poses_left\selection.csv
set POSE=%~1
if "%POSE%"=="" set POSE=pinch
if "%POSE%"=="pinch" goto pose_ok
if "%POSE%"=="open_palm" goto pose_ok
if "%POSE%"=="fist" goto pose_ok
if "%POSE%"=="peace" goto pose_ok
echo "%POSE%" is not a demo pose. Use one of: pinch, open_palm, fist, peace. Nothing was recorded.
pause
exit /b 2

:pose_ok
echo Professor demo, MOCK: %POSE%. NO HARDWARE: mock glove and mock camera, nothing real is measured.
echo.
echo Readiness check on the mock camera stream, 3 seconds:
%PY% scripts\leap\check_setup.py --mock --seconds 3
echo.
echo Press a key to run the mock recorder for %POSE%.
pause
rem --prep 0 --duration 1 (the real demo holds 5 s): the mock camera swaps hands every 4 s, so a
rem longer mock hold always loses the hand. Same flags as the ten-pose mock rehearsal (GATES_paired_pose G18).
%PY% scripts\record_simultaneous.py --camera leap --hand left --poses %POSE% --takes 1 --first-take auto --prep 0 --duration 1 --out-dir %REC% --mock-glove --mock-leap --no-pose-check --note "professor demo, mock rehearsal"
if errorlevel 1 echo The mock recorder reported a failure; the recorded figure below does not depend on it.
echo.
echo A MOCK TAKE HAS NO PHOTO, so the paired-pose tool refuses it: the live flow ends here, at the recorder.
echo Opening instead the PRE-RECORDED figure of %POSE% from the ten-pose session of 5 and 6 November 2026,
echo the take accepted by the selection rule:
set TAKE=
for /f "usebackq delims=" %%T in (`%PY% -c "import csv,sys;r=[x['accepted_take'] for x in csv.DictReader(open(sys.argv[1],encoding='utf-8')) if x['pose']==sys.argv[2] and x['hand']=='left'];print(r[0] if r else '')" %SEL% %POSE%`) do set TAKE=%%T
if "%TAKE%"=="" goto no_take
set FIG=results\paired_pose\ten_poses_left\%TAKE%\figure.png
if not exist "%FIG%" goto no_fig
echo FALLBACK FIGURE: %CD%\%FIG%
echo Left: the camera's photo. Middle: the Leap hand model. Right: the glove hand model.
start "" "%FIG%"
pause
exit /b 0

:no_take
echo %SEL% names no accepted take for %POSE%, so there is no recorded figure to open.
pause
exit /b 1

:no_fig
echo The accepted take %TAKE% has no figure.png in results\paired_pose\ten_poses_left\%TAKE%.
pause
exit /b 1
