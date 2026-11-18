@echo off
cd /d "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
echo Step 6. Gloves on + camera. Condition C, object with the hand_on_object hint: 15 takes.
echo Right hand on the table, outside the camera's view, except to hand over the object.
.venv\Scripts\python.exe scripts\pilot_run.py C
echo.
echo Finished. This window can be closed.
pause
