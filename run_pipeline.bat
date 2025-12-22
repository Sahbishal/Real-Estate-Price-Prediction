@echo off
echo Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo Running Training Pipeline...
python main.py
pause
