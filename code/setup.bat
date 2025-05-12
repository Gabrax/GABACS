@echo off
setlocal

if not exist ENV (
    echo Creating virtual environment...
    py -m venv ENV

    call ENV/Scripts/activate.bat

    py -m pip install --upgrade pip

    py -m pip install numpy opencv-contrib-python scikit-learn

    echo All packages installed successfully.

    py stereo.py
) else (

    call ENV/Scripts/activate.bat

    py stereo.py
)

