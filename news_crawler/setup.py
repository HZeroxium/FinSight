#!/usr/bin/env python3
"""
Setup script for News Crawler Service
This script helps with initial setup and configuration
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print(f"📋 {description}")
    try:
        result = subprocess.run(
            command, shell=True, check=check, capture_output=True, text=True
        )
        if result.stdout:
            print(f"✅ {result.stdout.strip()}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_virtual_environment():
    """Create virtual environment"""
    venv_path = Path(".venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True

    return run_command("python -m venv .venv", "Creating virtual environment")


def install_dependencies():
    """Install project dependencies"""
    # Determine activation script based on OS
    if os.name == "nt":  # Windows
        activate_script = ".venv\\Scripts\\activate"
        pip_command = ".venv\\Scripts\\pip"
    else:  # Unix-like
        activate_script = ".venv/bin/activate"
        pip_command = ".venv/bin/pip"

    commands = [
        (f"{pip_command} install --upgrade pip", "Upgrading pip"),
        (f"{pip_command} install -r requirements.txt", "Installing dependencies"),
    ]

    for command, description in commands:
        if not run_command(command, description):
            return False

    return True


def setup_environment_file():
    """Setup environment configuration file"""
    env_file = Path(".env")
    env_example = Path(".env.example")

    if env_file.exists():
        print("✅ .env file already exists")
        return True

    if env_example.exists():
        try:
            env_example.rename(env_file)
            print("✅ Created .env file from .env.example")
            print("⚠️  Please edit .env file with your actual configuration values")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    else:
        print("⚠️  .env.example file not found")
        return False


def create_directories():
    """Create necessary directories"""
    directories = ["logs", "data", "cache"]

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created {directory} directory")
        else:
            print(f"✅ {directory} directory already exists")

    return True


def validate_setup():
    """Validate the setup"""
    print("🔍 Validating setup...")

    # Check if virtual environment exists
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found")
        return False

    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False

    # Check if key directories exist
    for directory in ["logs", "data"]:
        if not Path(directory).exists():
            print(f"❌ {directory} directory not found")
            return False

    print("✅ Setup validation passed")
    return True


def display_next_steps():
    """Display next steps for the user"""
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your configuration:")
    print("   - Set TAVILY_API_KEY to your actual API key")
    print("   - Configure MongoDB and RabbitMQ URLs if needed")
    print("")
    print("2. Start the FastAPI server:")
    print("   python -m src.main")
    print("   or")
    print("   start_server.bat (Windows)")
    print("")
    print("3. Start the cron job service:")
    print("   python -m src.news_crawler_job start")
    print("   or")
    print("   manage_job.bat start (Windows)")
    print("")
    print("4. Access the API documentation:")
    print("   http://localhost:8000/docs")
    print("")
    print("5. Run tests:")
    print("   pytest test_integration.py")


def main():
    """Main setup function"""
    print("🚀 News Crawler Service Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Failed to create virtual environment")
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)

    # Setup environment file
    if not setup_environment_file():
        print("❌ Failed to setup environment file")
        sys.exit(1)

    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)

    # Validate setup
    if not validate_setup():
        print("❌ Setup validation failed")
        sys.exit(1)

    # Display next steps
    display_next_steps()


if __name__ == "__main__":
    main()
