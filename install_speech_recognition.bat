@echo off
echo تثبيت المكتبات اللازمة لتشغيل برنامج استخراج النصوص من الفيديو
echo ===================================================

python -m pip install --upgrade pip
pip install flask==2.0.1
pip install pysrt==1.1.2
pip install pytube==15.0.0
pip install moviepy==1.0.3
pip install pydub==0.25.1
pip install SpeechRecognition==3.8.1
pip install numpy==1.22.0

echo.
echo محاولة تثبيت مكتبة webrtcvad لتحسين دقة اكتشاف الكلام...
pip install webrtcvad==2.0.10

echo.
echo تثبيت مكتبة yt-dlp كبديل لتنزيل فيديوهات يوتيوب...
pip install yt-dlp==2025.2.19

echo.
echo تم الانتهاء من تثبيت المكتبات
echo اضغط أي مفتاح للخروج...
pause > nul
