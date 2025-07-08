@echo off
echo Starting News Crawler FastAPI Server...
echo.

REM Navigate to the project directory
cd /d "%~dp0"

REM Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo No virtual environment found, using global Python...
)

REM Install dependencies if requirements.txt exists
if exist "requirements.txt" (
    echo Installing/updating dependencies...
    pip install -r requirements.txt
    echo.
)

REM Set environment variables
set PYTHONPATH=%CD%
set TAVILY_API_KEY=your_tavily_api_key_here

REM Start the FastAPI server
echo Starting FastAPI server...
python -m src.main

REM Pause to keep the window open on error
pause
