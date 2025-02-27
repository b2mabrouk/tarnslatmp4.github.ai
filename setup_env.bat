@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install flask==2.0.1
pip install pytube==12.1.0
pip install numpy==1.23.5
pip install moviepy==1.0.3
pip install pysrt==1.1.2
pip install werkzeug==2.0.1
pip install gunicorn==20.1.0

echo Attempting to install faster-whisper...
pip install faster-whisper==0.10.0

echo Setup complete!
echo To activate the environment in the future, run: venv\Scripts\activate.bat
echo To run the application, use: python app.py
