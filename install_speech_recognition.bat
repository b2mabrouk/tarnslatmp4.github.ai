@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing SpeechRecognition and pydub...
pip install SpeechRecognition pydub

echo Installation complete!
echo To run the application, use: python app.py
