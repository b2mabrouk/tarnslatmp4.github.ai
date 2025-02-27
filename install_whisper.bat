@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing Whisper...
pip install openai-whisper

echo Installation complete!
echo To run the application, use: python app.py
