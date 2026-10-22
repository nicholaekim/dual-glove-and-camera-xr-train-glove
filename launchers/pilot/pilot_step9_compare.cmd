@echo off
cd /d "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
echo Step 9. No hardware. The comparison of the five runs, written into recordings\pilot_hoo\comparison.
.venv\Scripts\python.exe scripts\pilot_run.py compare
echo.
echo Finished. This window can be closed.
pause
