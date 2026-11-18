@echo off
rem Professor's live demo of the paired-pose pipeline: ONE pose, GLOVES ON + CAMERA, LEFT hand.
rem Usage: prof_demo.cmd POSE, one of pinch (the default), open_palm, fist, peace
rem Records one take into recordings\prof_demo (never the ten-pose session), runs the paired-pose
rem tool on that folder and opens the new take's figure.png: photo, Leap model, glove model.
rem Nothing is deleted: every run adds the next take number (--first-take auto).
setlocal
cd /d "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
set PY=.venv\Scripts\python.exe
set REC=recordings\prof_demo
set OUT=results\paired_pose\prof_demo
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
echo Professor demo: %POSE%. GLOVES ON + CAMERA, LEFT hand.
echo.
echo SETUP CHECKLIST:
echo  - Gloves calibrated ONCE in XR Trainer, streaming to port 9002. No recalibration during the demo.
echo  - Camera flat on the table, lenses up: its picture is dark, with no face or ceiling in it.
echo  - Right hand resting on the table, outside the camera's view.
echo  - Left hand 20 to 28 cm above the module.
echo  - Start from an OPEN PALM, then the pose. The camera window shows the target pose in its
echo    top-right corner: copy it, and hold it still while REC shows.
echo.
echo Readiness check, 3 seconds (camera only; hold the left hand open over the module):
%PY% scripts\leap\check_setup.py --seconds 3
if errorlevel 1 goto check_failed
echo.
echo Press a key to record %POSE%: open palm first, then the pose.
pause
call :newest BEFORE
%PY% scripts\record_simultaneous.py --camera leap --hand left --poses %POSE% --takes 1 --first-take auto --duration 5 --out-dir %REC% --note "professor live demo"
if errorlevel 1 goto rec_failed
call :newest AFTER
if "%AFTER%"=="%BEFORE%" goto no_new
echo.
echo Running the paired-pose tool on %REC%:
%PY% scripts\paired_pose_compare.py %REC%
set FIG=%OUT%\%AFTER%\figure.png
if not exist "%FIG%" goto no_fig
echo.
echo FIGURE: %CD%\%FIG%
echo Left: the camera's photo. Middle: the Leap hand model. Right: the glove hand model.
start "" "%FIG%"
pause
exit /b 0

:check_failed
echo.
echo The readiness check FAILED. Fix what it names, then run this launcher again. Nothing was recorded.
echo No hardware at all? Run prof_demo_mock.cmd: it shows the recorded figure of this pose.
pause
exit /b 1

:rec_failed
echo.
echo The recorder stopped and reported a failure. Nothing was deleted. Run this launcher again
echo (the next take number), or run prof_demo_mock.cmd to show the recorded figure of this pose.
pause
exit /b 1

:no_new
echo.
echo The recorder finished but no new %POSE% take is in %REC%\leap, so there is no new figure to open.
echo Nothing older is shown in its place. Run this launcher again, or run prof_demo_mock.cmd.
pause
exit /b 1

:no_fig
echo.
echo The tool made no figure for %AFTER%. Its row in %OUT%\comparison.csv says why (last column):
findstr /b /c:"%AFTER%" %OUT%\comparison.csv
echo The take stays on disk. Run this launcher again for the next take, or run prof_demo_mock.cmd.
pause
exit /b 1

rem Sets the variable named %1 to the stem of the newest %POSE% take in %REC%\leap (by its date and
rem time), or leaves it empty when there is none.
:newest
set %1=
for /f "usebackq delims=" %%S in (`%PY% -c "import pathlib,sys;ms=sorted(pathlib.Path(sys.argv[1]).glob(sys.argv[2]+'_left_take*.meta.json'),key=lambda p:p.name.split('.')[0].rsplit('_',2)[1:]);print(ms[-1].name.split('.')[0] if ms else '')" %REC%\leap %POSE%`) do set %1=%%S
exit /b 0
