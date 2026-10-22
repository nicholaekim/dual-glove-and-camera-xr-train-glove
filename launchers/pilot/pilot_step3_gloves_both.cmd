@echo off
cd /d "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"
echo Step 3. Gloves on + camera: put the gloves on, calibrate once in XR Trainer, then this records the glove setup for the whole pilot (both gloves connected).
echo It records nothing else and shows the step 4 command.
.venv\Scripts\python.exe scripts\pilot_run.py A --gloves both --print
echo.
echo Finished. This window can be closed.
pause
