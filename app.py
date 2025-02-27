import os
import tempfile
import json
import uuid
import threading
import time
from datetime import timedelta
from flask import Flask, request, jsonify, send_from_directory
import pysrt
import logging
import shutil
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.playback import play
from pydub.generators import WhiteNoise
from pydub import effects

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')

# Ensure upload directory exists
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'video_subtitle_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flag to track if speech recognition is available
speech_recognition_available = False
recognizer = None

# Dictionary to store task progress
tasks = {}

# Try to import optional dependencies
try:
    from moviepy.editor import VideoFileClip
    from pytube import YouTube
    import speech_recognition as sr
    
    # Initialize speech recognition
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300  # Increase energy threshold for better recognition
    recognizer.dynamic_energy_threshold = True  # Enable dynamic energy threshold
    recognizer.pause_threshold = 0.8  # Shorter pause threshold for more accurate sentence breaks
    
    speech_recognition_available = True
    logger.info("Speech recognition initialized successfully")
except ImportError as e:
    logger.warning(f"Some dependencies could not be imported: {e}")
    logger.warning("Running in fallback mode with sample subtitles only")

# Try to import advanced voice activity detection
vad_available = False
try:
    import webrtcvad
    vad = webrtcvad.Vad(3)  # Aggressiveness mode 3 (most aggressive)
    vad_available = True
    logger.info("WebRTC VAD initialized successfully")
except ImportError as e:
    logger.warning(f"WebRTC VAD not available: {e}")

def process_with_vad(audio_segment, frame_duration_ms=30, padding_duration_ms=300):
    """Process audio with WebRTC Voice Activity Detection to improve speech recognition"""
    if not vad_available:
        logger.warning("WebRTC VAD not available, skipping VAD processing")
        return audio_segment
    
    try:
        # Convert audio to the format required by WebRTC VAD (16-bit PCM, mono)
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
        
        if audio_segment.frame_rate != 16000:
            audio_segment = audio_segment.set_frame_rate(16000)
        
        # Create a VAD instance with aggressiveness level 3 (most aggressive)
        vad = webrtcvad.Vad(3)
        
        # Get raw PCM data
        raw_data = audio_segment.raw_data
        
        # Calculate frame size
        frame_size = int(audio_segment.frame_rate * frame_duration_ms / 1000)
        
        # Process frames
        voiced_frames = []
        for i in range(0, len(raw_data) - frame_size, frame_size):
            frame = raw_data[i:i + frame_size]
            if len(frame) < frame_size:
                break
                
            is_speech = vad.is_speech(frame, audio_segment.frame_rate)
            if is_speech:
                voiced_frames.append(frame)
        
        # If no voiced frames were detected, return the original audio
        if not voiced_frames:
            logger.warning("No voiced frames detected, returning original audio")
            return audio_segment
        
        # Combine voiced frames with padding
        padding_size = int(audio_segment.frame_rate * padding_duration_ms / 1000)
        processed_data = b''.join(voiced_frames)
        
        # Create a new AudioSegment from the processed data
        processed_segment = AudioSegment(
            data=processed_data,
            sample_width=audio_segment.sample_width,
            frame_rate=audio_segment.frame_rate,
            channels=1
        )
        
        logger.info(f"VAD processing complete: reduced audio from {len(audio_segment)} ms to {len(processed_segment)} ms")
        return processed_segment
    except Exception as e:
        logger.error(f"Error in VAD processing: {str(e)}")
        return audio_segment

def split_on_sentence_breaks(audio_segment, min_silence_len=500, silence_thresh=-40, keep_silence=300):
    """Split audio into smaller chunks based on sentence breaks (silence detection)"""
    try:
        # First try with provided parameters
        chunks = split_on_silence(
            audio_segment, 
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )
        
        # If we got too few chunks, try with less strict parameters
        if len(chunks) < 5 and len(audio_segment) > 10000:  # If less than 5 chunks and audio longer than 10 seconds
            logger.info("Too few sentence breaks detected, adjusting parameters")
            silence_thresh = silence_thresh + 5  # Less strict silence threshold
            min_silence_len = max(300, min_silence_len - 100)  # Shorter minimum silence
            
            chunks = split_on_silence(
                audio_segment, 
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=keep_silence
            )
        
        # If we still have too few chunks, try even less strict parameters
        if len(chunks) < 10 and len(audio_segment) > 20000:  # If less than 10 chunks and audio longer than 20 seconds
            logger.info("Still too few sentence breaks, using more aggressive parameters")
            silence_thresh = silence_thresh + 5  # Even less strict
            min_silence_len = max(200, min_silence_len - 100)  # Even shorter minimum silence
            
            chunks = split_on_silence(
                audio_segment, 
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=keep_silence
            )
            
        # If we still have too few chunks and the audio is long, try with very aggressive parameters
        if len(chunks) < 20 and len(audio_segment) > 60000:  # If less than 20 chunks and audio longer than 1 minute
            logger.info("Still not enough sentence breaks for long audio, using very aggressive parameters")
            silence_thresh = -30  # Very permissive silence threshold
            min_silence_len = 150  # Very short silence is considered a break
            
            chunks = split_on_silence(
                audio_segment, 
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=200
            )
        
        logger.info(f"Split audio into {len(chunks)} sentence chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error in sentence break detection: {str(e)}")
        return []

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

def process_youtube_video(task_id, youtube_url, language):
    """Process YouTube video in a background thread"""
    try:
        logger.info(f"Starting to process YouTube URL: {youtube_url}, task_id: {task_id}")
        
        # Update task status
        tasks[task_id]['message'] = 'جاري تنزيل الفيديو من يوتيوب...'
        tasks[task_id]['progress'] = 5
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Download YouTube video
        yt = YouTube(youtube_url)
        video_title = yt.title
        
        # Get the first stream that's progressive and has both video and audio
        stream = yt.streams.filter(progressive=True).first()
        
        if not stream:
            logger.warning("No progressive stream found, trying video-only stream")
            stream = yt.streams.filter(only_video=True).first()
            
        if not stream:
            raise Exception("No suitable video stream found")
            
        video_path = stream.download(output_path=temp_dir)
        
        logger.info(f"Downloaded YouTube video to: {video_path}")
        
        # Update task status
        tasks[task_id]['message'] = 'تم تنزيل الفيديو، جاري استخراج الترجمة...'
        tasks[task_id]['progress'] = 20
        
        # Generate subtitles
        subtitles = generate_subtitles_with_speech_recognition(video_path, language, task_id)
        
        # Update task status
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['progress'] = 100
        tasks[task_id]['message'] = 'تم إنشاء الترجمة بنجاح'
        tasks[task_id]['result'] = {
            'srt_content': subtitles,
            'filename': f"{video_title}.srt"
        }
        
        logger.info(f"Completed processing YouTube task {task_id}")
        
        # Clean up
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Removed temporary YouTube video file: {video_path}")
            
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
                logger.info(f"Removed temporary YouTube directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary YouTube files: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error processing YouTube video: {str(e)}")
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['message'] = f'حدث خطأ: {str(e)}'
        tasks[task_id]['error'] = str(e)

@app.route('/progress/<task_id>')
def get_progress(task_id):
    """Get the progress of a task"""
    try:
        logger.info(f"Checking progress for task: {task_id}")
        
        if task_id not in tasks:
            logger.warning(f"Task ID not found: {task_id}")
            return jsonify({'error': 'Task not found'}), 404
        
        task = tasks[task_id]
        
        # If task is completed, return the result
        if task['status'] == 'completed':
            logger.info(f"Task {task_id} is completed, returning result")
            return jsonify({
                'status': 'completed',
                'progress': 100,
                'message': task.get('message', 'تم إنشاء الترجمة بنجاح'),
                'result': task.get('result', {})
            })
        
        # If task is in error state, return the error
        if task['status'] == 'error':
            logger.warning(f"Task {task_id} is in error state: {task.get('error', 'Unknown error')}")
            return jsonify({
                'status': 'error',
                'message': task.get('message', 'حدث خطأ أثناء المعالجة'),
                'error': task.get('error', 'Unknown error')
            })
        
        # Task is still processing
        logger.info(f"Task {task_id} is still processing: {task.get('progress', 0)}% - {task.get('message', 'جاري المعالجة...')}")
        return jsonify({
            'status': 'processing',
            'progress': task.get('progress', 0),
            'message': task.get('message', 'جاري المعالجة...')
        })
        
    except Exception as e:
        logger.error(f"Error checking progress for task {task_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_video():
    try:
        logger.info("Received process request")
        
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            'status': 'processing',
            'message': 'بدء معالجة الفيديو...',
            'progress': 0
        }
        
        logger.info(f"Created task with ID: {task_id}")
        
        if request.content_type and 'application/json' in request.content_type:
            # Handle YouTube URL
            data = request.json
            youtube_url = data.get('youtube_url')
            language = data.get('language')
            
            logger.info(f"Processing YouTube URL: {youtube_url}, Language: {language}")
            
            if not youtube_url or not language:
                return jsonify({'error': 'Missing YouTube URL or language'}), 400
            
            if not speech_recognition_available:
                logger.warning("Speech recognition not available, returning sample subtitles")
                subtitles = generate_sample_subtitles(language)
                return jsonify({
                    'subtitles': subtitles,
                    'warning': 'Using sample subtitles because speech recognition is not available'
                })
                
            # Process YouTube video in a background thread
            threading.Thread(target=process_youtube_video, args=(task_id, youtube_url, language)).start()
            
            logger.info(f"Started background thread for YouTube processing with task ID: {task_id}")
            
            return jsonify({
                'task_id': task_id,
                'message': 'جاري معالجة الفيديو في الخلفية...'
            })
            
        else:
            # Handle file upload
            if 'video' not in request.files:
                return jsonify({'error': 'No video file uploaded'}), 400
                
            file = request.files['video']
            language = request.form.get('language')
            
            logger.info(f"Processing uploaded file: {file.filename}, Language: {language}")
            
            if not file or not language:
                return jsonify({'error': 'Missing file or language'}), 400
            
            if not speech_recognition_available:
                logger.warning("Speech recognition not available, returning sample subtitles")
                subtitles = generate_sample_subtitles(language)
                return jsonify({
                    'subtitles': subtitles,
                    'warning': 'Using sample subtitles because speech recognition is not available'
                })
            
            # Save uploaded file temporarily before processing in a thread
            temp_dir = tempfile.mkdtemp()
            filename = file.filename
            video_path = os.path.join(temp_dir, filename)
            file.save(video_path)
            
            logger.info(f"Saved uploaded video to: {video_path}")
            
            # Process uploaded file in a background thread
            threading.Thread(
                target=process_uploaded_file,
                args=(task_id, video_path, filename, language)
            ).start()
            
            logger.info(f"Started background thread for file processing with task ID: {task_id}")
            
            return jsonify({
                'task_id': task_id,
                'message': 'جاري معالجة الفيديو في الخلفية...'
            })
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_uploaded_file(task_id, video_path, filename, language):
    """Process an uploaded video file in a background thread"""
    try:
        logger.info(f"Starting to process uploaded file: {filename}, task_id: {task_id}")
        
        # Update task status
        tasks[task_id]['message'] = 'بدء معالجة الملف المرفوع...'
        tasks[task_id]['progress'] = 5
        
        # Generate subtitles
        subtitles = generate_subtitles_with_speech_recognition(video_path, language, task_id)
        
        # Update task status
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['progress'] = 100
        tasks[task_id]['message'] = 'تم إنشاء الترجمة بنجاح'
        tasks[task_id]['result'] = {
            'srt_content': subtitles,
            'filename': f"{os.path.splitext(filename)[0]}.srt"
        }
        
        logger.info(f"Completed processing task {task_id}")
        
        # Clean up
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Removed temporary video file: {video_path}")
            
            temp_dir = os.path.dirname(video_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
                logger.info(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['message'] = f'حدث خطأ: {str(e)}'
        tasks[task_id]['error'] = str(e)

def generate_subtitles_with_speech_recognition(video_path, language, task_id):
    """Extract audio from video and generate subtitles using speech recognition"""
    logger.info(f"Generating subtitles for: {video_path}")
    
    if not speech_recognition_available:
        logger.warning("Speech recognition is not available, returning sample subtitles")
        return generate_sample_subtitles(language)
    
    try:
        # Extract audio from video
        video = VideoFileClip(video_path)
        total_duration_ms = int(video.duration * 1000)  # Total duration in milliseconds
        logger.info(f"Video duration: {format_timestamp(total_duration_ms)} ({total_duration_ms} ms)")
        
        audio_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{os.getpid()}.wav")
        
        # Update task status
        tasks[task_id]['message'] = 'جاري استخراج الصوت من الفيديو...'
        
        # Extract audio with higher quality settings
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', ffmpeg_params=["-ac", "1", "-ar", "44100"])
        
        logger.info(f"Extracted audio to: {audio_path}")
        
        # Update task status
        tasks[task_id]['message'] = 'تم استخراج الصوت، جاري تحليل الكلام...'
        tasks[task_id]['progress'] = 10
        
        # Map language code to Google Speech Recognition language code
        language_map = {
            'ar': 'ar-SA',  # Arabic (Saudi Arabia)
            'en': 'en-US',  # English (US)
            'tr': 'tr-TR',  # Turkish
            'fr': 'fr-FR',  # French
            'es': 'es-ES',  # Spanish
            'de': 'de-DE'   # German
        }
        
        speech_language = language_map.get(language, language)
        
        # Create SRT file content
        srt_content = ""
        subtitle_index = 1
        
        # Load audio with pydub for processing
        audio_segment = AudioSegment.from_wav(audio_path)
        audio_duration_ms = len(audio_segment)
        logger.info(f"Audio duration: {format_timestamp(audio_duration_ms)} ({audio_duration_ms} ms)")
        
        # Normalize audio to improve speech recognition
        audio_segment = normalize_audio(audio_segment)
        
        # Apply VAD to improve speech detection if available and audio is longer than 10 seconds
        if vad_available and audio_duration_ms > 10000:
            audio_segment = process_with_vad(audio_segment)
        
        # Try to detect silence for better chunking
        logger.info("Attempting to detect sentence breaks for optimal chunking")
        chunks = split_on_sentence_breaks(audio_segment)
        
        # Create a temporary directory to store the audio chunks
        temp_chunk_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory for chunks: {temp_chunk_dir}")
        
        # Calculate and store the start position of each chunk
        chunk_positions = []
        current_position = 0
        
        for chunk in chunks:
            chunk_positions.append(current_position)
            current_position += len(chunk)
        
        # If no chunks were created or error occurred, use time-based chunking as fallback
        if len(chunks) <= 1:
            logger.info("Sentence break detection produced too few chunks, falling back to time-based chunking")
            chunk_length_ms = 2000  # 2 seconds per chunk for better sentence detection
            chunks = [audio_segment[i:i+chunk_length_ms] for i in range(0, len(audio_segment), chunk_length_ms)]
            
            # Recalculate chunk positions
            chunk_positions = []
            current_position = 0
            for chunk in chunks:
                chunk_positions.append(current_position)
                current_position += len(chunk)
                
            logger.info(f"Created {len(chunks)} time-based chunks of {chunk_length_ms/1000} seconds each")
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Export chunk to a temporary WAV file
            chunk_path = os.path.join(temp_chunk_dir, f"chunk_{i}.wav")
            chunk.export(chunk_path, format="wav", parameters=["-ac", "1", "-ar", "44100"])
            
            # Update task progress
            progress = 10 + int((i / len(chunks)) * 80)
            tasks[task_id]['progress'] = progress
            tasks[task_id]['message'] = f"جاري معالجة المقطع {i+1} من {len(chunks)} ({progress}%)"
            logger.info(f"Processing chunk {i+1}/{len(chunks)} - Progress: {progress}%")
            
            # Calculate precise start and end time for this chunk
            start_ms = chunk_positions[i]
            end_ms = start_ms + len(chunk)
            
            # Ensure we don't exceed the total duration
            if end_ms > audio_duration_ms:
                end_ms = audio_duration_ms
            
            # Format times for SRT with precise milliseconds
            start_time_srt = format_timestamp(start_ms)
            end_time_srt = format_timestamp(end_ms)
            
            logger.info(f"Chunk {i+1} timing: {start_time_srt} --> {end_time_srt}")
            
            # Try multiple recognition attempts with different settings
            text = ""
            recognition_attempts = 0
            max_attempts = 2
            
            while not text and recognition_attempts < max_attempts:
                try:
                    with sr.AudioFile(chunk_path) as source:
                        # Adjust for noise if this is a retry
                        if recognition_attempts > 0:
                            logger.info(f"Attempt {recognition_attempts+1} for chunk {i+1} with noise adjustment")
                            audio_data = recognizer.record(source)
                            recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        else:
                            audio_data = recognizer.record(source)
                        
                        # Try with higher phrase threshold for better results
                        if recognition_attempts > 0:
                            text = recognizer.recognize_google(audio_data, language=speech_language, 
                                                             show_all=False)
                        else:
                            text = recognizer.recognize_google(audio_data, language=speech_language)
                        
                        if text.strip():
                            logger.info(f"Successfully recognized text in chunk {i+1} on attempt {recognition_attempts+1}")
                            break
                except sr.UnknownValueError:
                    logger.warning(f"Speech recognition could not understand audio in chunk {i+1}, attempt {recognition_attempts+1}")
                    recognition_attempts += 1
                except sr.RequestError as e:
                    logger.error(f"Error with speech recognition request: {e}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error processing chunk {i+1}: {str(e)}")
                    break
            
            # Add subtitle if text was recognized
            if text.strip():
                srt_content += f"{subtitle_index}\n"
                srt_content += f"{start_time_srt} --> {end_time_srt}\n"
                srt_content += f"{text.strip()}\n\n"
                subtitle_index += 1
                logger.info(f"Added subtitle segment {subtitle_index-1}: {text[:30]}...")
            
            # Remove temporary chunk file
            try:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary chunk file {chunk_path}: {e}")
        
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_chunk_dir, ignore_errors=True)
            logger.info(f"Removed temporary chunk directory: {temp_chunk_dir}")
        except Exception as e:
            logger.warning(f"Could not remove temporary chunk directory {temp_chunk_dir}: {e}")
        
        # Clean up temporary audio file
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Removed temporary audio file: {audio_path}")
        except Exception as e:
            logger.warning(f"Could not remove temporary audio file {audio_path}: {e}")
        
        # If no subtitles were generated, return sample subtitles
        if subtitle_index == 1:
            logger.warning("No speech detected in audio, returning sample subtitles")
            return generate_sample_subtitles(language)
        
        logger.info(f"Generated {subtitle_index-1} subtitles")
        
        # Update task status
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['message'] = 'تم إنشاء الترجمة بنجاح'
        
        return srt_content
    except Exception as e:
        logger.error(f"Error in generate_subtitles_with_speech_recognition: {str(e)}")
        return generate_sample_subtitles(language)

def normalize_audio(audio_segment):
    """Normalize audio to improve speech recognition"""
    try:
        # Normalize to -20dB
        normalized_audio = effects.normalize(audio_segment, headroom=5.0)
        logger.info("Audio normalized successfully")
        return normalized_audio
    except Exception as e:
        logger.warning(f"Could not normalize audio: {e}")
        return audio_segment

def format_time(td):
    """Format timedelta object to SRT time format (HH:MM:SS,mmm)"""
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def format_timestamp(milliseconds):
    """Format milliseconds to SRT time format (HH:MM:SS,mmm)"""
    total_seconds = milliseconds // 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    ms = milliseconds % 1000
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"

def generate_sample_subtitles(language):
    """Generate sample subtitles based on selected language (fallback)"""
    if language == 'ar':
        return """1
00:00:00,000 --> 00:00:05,000
مرحبا بكم في خدمة استخراج الترجمة

2
00:00:05,500 --> 00:00:10,000
نعتذر، لم نتمكن من استخراج الترجمة من هذا الفيديو

3
00:00:10,500 --> 00:00:15,000
يرجى التأكد من أن الفيديو يحتوي على كلام واضح

4
00:00:15,500 --> 00:00:20,000
يمكنك المحاولة مرة أخرى مع فيديو آخر
"""
    elif language == 'tr':
        return """1
00:00:00,000 --> 00:00:05,000
Altyazı çıkarma hizmetine hoş geldiniz

2
00:00:05,500 --> 00:00:10,000
Üzgünüz, bu videodan altyazı çıkaramadık

3
00:00:10,500 --> 00:00:15,000
Lütfen videonun net konuşma içerdiğinden emin olun

4
00:00:15,500 --> 00:00:20,000
Başka bir video ile tekrar deneyebilirsiniz
"""
    else:
        return """1
00:00:00,000 --> 00:00:05,000
Welcome to the subtitle extraction service

2
00:00:05,500 --> 00:00:10,000
Sorry, we couldn't extract subtitles from this video

3
00:00:10,500 --> 00:00:15,000
Please make sure the video contains clear speech

4
00:00:15,500 --> 00:00:20,000
You can try again with another video
"""

@app.route('/setup_info')
def setup_info():
    """Return information about the setup and available features"""
    info = {
        "speech_recognition_available": speech_recognition_available,
        "vad_available": vad_available,
        "supported_languages": [
            {"code": "ar", "name": "العربية (Arabic)"},
            {"code": "en", "name": "الإنجليزية (English)"},
            {"code": "tr", "name": "التركية (Turkish)"},
            {"code": "fr", "name": "الفرنسية (French)"},
            {"code": "es", "name": "الإسبانية (Spanish)"},
            {"code": "de", "name": "الألمانية (German)"}
        ],
        "version": "2.0.0",
        "features": [
            "Multilingual speech recognition",
            "Silence-based audio chunking",
            "Voice activity detection (VAD)",
            "Audio normalization",
            "YouTube video processing",
            "Local video file processing"
        ]
    }
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
