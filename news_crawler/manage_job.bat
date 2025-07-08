@echo off
echo News Crawler Job Management Script
echo ====================================
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

REM Set environment variables
set PYTHONPATH=%CD%
set TAVILY_API_KEY=your_tavily_api_key_here

REM Parse command line arguments
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=help

if "%COMMAND%"=="help" goto :help
if "%COMMAND%"=="start" goto :start
if "%COMMAND%"=="stop" goto :stop
if "%COMMAND%"=="status" goto :status
if "%COMMAND%"=="run" goto :run
if "%COMMAND%"=="config" goto :config
goto :help

:help
echo Usage: manage_job.bat [command]
echo.
echo Commands:
echo   start   - Start the news crawler cron job service
echo   stop    - Stop the news crawler cron job service
echo   status  - Check the status of the job service
echo   run     - Run a manual news crawl job
echo   config  - Display current configuration
echo   help    - Show this help message
echo.
echo Examples:
echo   manage_job.bat start
echo   manage_job.bat status
echo   manage_job.bat run
echo   manage_job.bat stop
goto :end

:start
echo Starting News Crawler Job Service...
python -m src.news_crawler_job start
goto :end

:stop
echo Stopping News Crawler Job Service...
python -m src.news_crawler_job stop
goto :end

:status
echo Checking News Crawler Job Service status...
python -m src.news_crawler_job status
goto :end

:run
echo Running manual news crawl...
python -m src.news_crawler_job run
goto :end

:config
echo Displaying current configuration...
python -m src.news_crawler_job config
goto :end

:end
echo.
echo Press any key to exit...
pause >nul
