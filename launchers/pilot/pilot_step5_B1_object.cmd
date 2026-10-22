@echo off
cd /d "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
echo Step 5. Gloves on + camera. Condition B1, object: 9 takes with the three objects of objects.txt.
echo Right hand on the table, outside the camera's view, except to hand over the object.
.venv\Scripts\python.exe scripts\pilot_run.py B1
echo.
echo Finished. This window can be closed.
pause
