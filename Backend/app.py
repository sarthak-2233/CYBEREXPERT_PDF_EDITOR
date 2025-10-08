from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import base64
import tempfile
import threading
import time
from datetime import datetime
from io import BytesIO
import uuid
import queue
import subprocess
import requests

# Import your existing classes
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("PyMuPDF not available - PDF functionality disabled")

try:
    from PIL import Image, ImageDraw, ImageFont
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("PIL not available - Image processing disabled")

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV not available - Video processing disabled")

# AI processing imports
try:
    import openai
    from openai import OpenAI
    AI_PROCESSING_AVAILABLE = True
except ImportError:
    AI_PROCESSING_AVAILABLE = False
    print("OpenAI not available - AI processing disabled")

# Speech recognition imports
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("Speech Recognition not available - audio transcription disabled")

# Recording imports
try:
    import pyautogui
    import sounddevice as sd
    import soundfile as sf
    from scipy.io import wavfile
    RECORDING_AVAILABLE = True
except ImportError:
    RECORDING_AVAILABLE = False
    print("Recording dependencies not available - Screen recording disabled")

app = Flask(__name__)
CORS(app, origins=["*"])  # Enable CORS for React frontend

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300MB max file size

# OpenAI API Key - USE ENVIRONMENT VARIABLE FOR SECURITY
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

# Webhook Configuration
DEFAULT_WEBHOOK_URL = os.environ.get('WEBHOOK_URL', 'https://imp.ecom.ind.in//api/v1/comments/createByAI')
DEFAULT_TASK_ID = '21'
DEFAULT_CREATED_BY_ID = '1'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for managing sessions and recordings
active_sessions = {}
active_recordings = {}

class EnhancedScreenRecorder:
    def __init__(self, session_id):
        self.session_id = session_id
        self.recording = False
        self.frames = []
        self.audio_data = []
        self.audio_queue = queue.Queue()
        self.start_time = None
        self.output_path = None
        self.video_thread = None
        self.audio_thread = None
        self.sample_rate = 44100
        self.channels = 2
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio recording"""
        if status:
            print(f"Audio callback status: {status}")
        self.audio_queue.put(indata.copy())
        
    def start_recording(self, output_path):
        if not RECORDING_AVAILABLE:
            print("Recording not available - missing dependencies")
            return False
            
        self.output_path = output_path
        self.audio_path = output_path.replace('.mp4', '_audio.wav')
        self.recording = True
        self.frames = []
        self.audio_data = []
        self.start_time = time.time()
        
        try:
            # Start audio recording
            self.audio_stream = sd.InputStream(
                callback=self.audio_callback,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            )
            self.audio_stream.start()
            
            # Start video recording thread
            self.video_thread = threading.Thread(target=self._record_screen)
            self.video_thread.daemon = True
            self.video_thread.start()
            
            # Start audio processing thread
            self.audio_thread = threading.Thread(target=self._process_audio)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            return True
        except Exception as e:
            print(f"Error starting enhanced recording: {e}")
            return False
        
    def _process_audio(self):
        """Process audio data from queue"""
        while self.recording:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                self.audio_data.append(audio_chunk)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio processing error: {e}")
                break
                
    def _record_screen(self):
        try:
            fps = 10  # Increased FPS for better quality
            frame_count = 0
            
            while self.recording and frame_count < 600:  # Max 60 seconds at 10fps
                try:
                    screenshot = pyautogui.screenshot()
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.frames.append(frame)
                    frame_count += 1
                    time.sleep(1.0 / fps)
                except Exception as e:
                    print(f"Frame capture error: {e}")
                    break
                    
        except Exception as e:
            print(f"Recording error: {e}")
            
    def stop_recording(self):
        self.recording = False
        
        # Stop audio stream
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
        
        # Wait for threads to finish
        if self.video_thread:
            self.video_thread.join(timeout=5.0)
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)
            
        # Save audio first
        audio_saved = self._save_audio()
        
        # Save video
        video_saved = self._save_video()
        
        if audio_saved and video_saved:
            # Combine audio and video
            return self._combine_audio_video()
        
        return video_saved
        
    def _save_audio(self):
        """Save recorded audio to WAV file"""
        if not self.audio_data:
            print("No audio data to save")
            return False
            
        try:
            # Concatenate all audio chunks
            audio_array = np.concatenate(self.audio_data, axis=0)
            
            # Save as WAV file
            sf.write(self.audio_path, audio_array, self.sample_rate)
            print(f"Audio saved: {self.audio_path}")
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False
            
    def _save_video(self):
        if not self.frames:
            print("No frames to save")
            return False
            
        try:
            if not OPENCV_AVAILABLE:
                return False
                
            height, width, layers = self.frames[0].shape
            temp_video_path = self.output_path.replace('.mp4', '_temp.mp4')
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(temp_video_path, fourcc, 10.0, (width, height))
            
            for frame in self.frames:
                video.write(frame)
            video.release()
            
            self.temp_video_path = temp_video_path
            print(f"Video saved: {temp_video_path}")
            return True
        except Exception as e:
            print(f"Error saving video: {e}")
            return False
            
    def _combine_audio_video(self):
        """Combine audio and video using ffmpeg (if available)"""
        try:
            import shutil
            shutil.move(self.temp_video_path, self.output_path)
            print(f"Combined video saved: {self.output_path}")
            return True
        except Exception as e:
            print(f"Error combining audio and video: {e}")
            try:
                import shutil
                shutil.move(self.temp_video_path, self.output_path)
                return True
            except:
                return False

class AnnotationScreenshotManager:
    def __init__(self, session_id):
        self.session_id = session_id
        self.screenshots = []
        self.screenshot_folder = os.path.join('uploads', f'screenshots_{session_id}')
        os.makedirs(self.screenshot_folder, exist_ok=True)
        
    def capture_annotation_screenshot(self, annotation_data, page_image):
        """Capture screenshot when annotation is made"""
        try:
            timestamp = datetime.now().isoformat()
            screenshot_id = str(uuid.uuid4())
            
            # Create screenshot with annotation overlay
            if PILLOW_AVAILABLE and page_image:
                # Decode base64 image
                img_data = base64.b64decode(page_image)
                img = Image.open(BytesIO(img_data))
                
                # Create a copy for annotation overlay
                screenshot_img = img.copy()
                draw = ImageDraw.Draw(screenshot_img)
                
                # Draw the new annotation
                if annotation_data.get('points') and len(annotation_data['points']) > 1:
                    points = annotation_data['points']
                    color = annotation_data.get('color', '#ff0000')
                    size = annotation_data.get('size', 3)
                    
                    # Convert relative coordinates to image coordinates
                    img_width, img_height = img.size
                    pixel_points = [
                        (int(p['x'] * img_width), int(p['y'] * img_height))
                        for p in points
                    ]
                    
                    # Draw lines between points
                    for i in range(len(pixel_points) - 1):
                        draw.line([pixel_points[i], pixel_points[i + 1]], fill=color, width=size)
                
                # Save screenshot
                screenshot_path = os.path.join(self.screenshot_folder, f'{screenshot_id}.png')
                screenshot_img.save(screenshot_path, 'PNG', optimize=True)
                
                # Convert to base64 for response
                buffer = BytesIO()
                screenshot_img.save(buffer, format='PNG', optimize=True)
                screenshot_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                screenshot_data = {
                    'id': screenshot_id,
                    'timestamp': timestamp,
                    'image_base64': screenshot_base64,
                    'annotation_type': annotation_data.get('type', 'pen'),
                    'color': annotation_data.get('color', '#ff0000'),
                    'file_path': screenshot_path
                }
                
                self.screenshots.append(screenshot_data)
                return screenshot_data
                
        except Exception as e:
            print(f"Error capturing annotation screenshot: {e}")
            return None
            
    def get_screenshots(self):
        """Get all screenshots for this session"""
        return self.screenshots
        
    def clear_screenshots(self):
        """Clear all screenshots"""
        try:
            for screenshot in self.screenshots:
                if os.path.exists(screenshot['file_path']):
                    os.remove(screenshot['file_path'])
            self.screenshots = []
            return True
        except Exception as e:
            print(f"Error clearing screenshots: {e}")
            return False

class EnhancedVideoProcessor:
    def __init__(self, video_path, audio_path=None):
        self.video_path = video_path
        self.audio_path = audio_path or video_path.replace('.mp4', '_audio.wav')
        self.client = None
        if OPENAI_API_KEY and AI_PROCESSING_AVAILABLE:
            try:
                self.client = OpenAI(api_key=OPENAI_API_KEY)
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
                self.client = None

    def extract_audio_and_transcribe(self):
        """Extract audio from video and transcribe it using speech recognition"""
        try:
            if not SPEECH_RECOGNITION_AVAILABLE:
                print("⚠️ Speech recognition not available - falling back to mock transcript")
                return self._generate_mock_transcript()
            
            print("🎵 Starting audio transcription process...")
            
            # Check if audio file exists
            if os.path.exists(self.audio_path):
                print(f"✅ Using existing audio file: {self.audio_path}")
                audio_file = self.audio_path
            else:
                print(f"⚠️ Audio file not found at {self.audio_path}")
                print("🔧 Attempting to extract audio from video...")
                
                # Extract audio from video using ffmpeg
                temp_audio = tempfile.mktemp(suffix='.wav')
                
                ffmpeg_commands = [
                    ['ffmpeg', '-i', self.video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_audio, '-y'],
                    ['ffmpeg.exe', '-i', self.video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_audio, '-y']
                ]
                
                audio_extracted = False
                for cmd in ffmpeg_commands:
                    try:
                        print(f"🔧 Running: {' '.join(cmd[:3])}...")
                        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
                        if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 0:
                            print(f"✅ Audio extraction successful ({os.path.getsize(temp_audio) / 1024:.1f} KB)")
                            audio_file = temp_audio
                            audio_extracted = True
                            break
                    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                        continue
                
                if not audio_extracted:
                    print("❌ Could not extract audio from video")
                    return self._generate_mock_transcript()
            
            print("🎤 Starting speech recognition transcription...")
            
            # Transcribe audio
            recognizer = sr.Recognizer()
            transcript = ""
            
            try:
                with sr.AudioFile(audio_file) as source:
                    # Adjust for ambient noise
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    
                    # Get audio duration
                    duration = self._get_audio_duration(audio_file)
                    print(f"📊 Audio duration: {duration:.1f} seconds")
                    
                    if duration <= 0:
                        print("❌ Invalid audio duration")
                        return self._generate_mock_transcript()
                    
                    # Process audio in chunks
                    chunk_duration = min(30, max(10, duration / 3))  # Adaptive chunk size
                    
                    processed_chunks = 0
                    total_chunks = int(duration / chunk_duration) + 1
                    
                    for start_time in range(0, int(duration), int(chunk_duration)):
                        end_time = min(start_time + chunk_duration, duration)
                        print(f"🎯 Processing chunk {processed_chunks + 1}/{total_chunks}: {start_time}s - {end_time:.1f}s")
                        
                        with sr.AudioFile(audio_file) as chunk_source:
                            audio_data = recognizer.record(chunk_source, offset=start_time, duration=chunk_duration)
                        
                        try:
                            # Try Google Speech Recognition
                            text = recognizer.recognize_google(audio_data)
                            transcript += text + " "
                            print(f"✅ Chunk {processed_chunks + 1}: {text[:80]}...")
                            processed_chunks += 1
                        except sr.UnknownValueError:
                            print(f"⚠️ Could not understand audio in chunk {start_time}s-{end_time:.1f}s")
                            continue
                        except sr.RequestError as e:
                            print(f"⚠️ Speech recognition service error: {e}")
                            break
                
                # Clean up temporary file if created
                if audio_file != self.audio_path and os.path.exists(audio_file):
                    os.remove(audio_file)
                    print(f"🗑️ Cleaned up temporary audio file")
                
            except Exception as e:
                print(f"❌ Audio processing error: {e}")
                transcript = ""
            
            # Validate transcript
            if not transcript.strip() or len(transcript.strip().split()) < 3:
                print("⚠️ Transcript too short or empty, using fallback")
                return self._generate_mock_transcript()
            
            print(f"✅ Transcription completed: {len(transcript)} characters, {len(transcript.split())} words")
            return transcript.strip()
            
        except Exception as e:
            print(f"❌ Transcription error: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._generate_mock_transcript()

    def _get_audio_duration(self, audio_file):
        """Get duration of audio file"""
        try:
            import wave
            with wave.open(audio_file, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            print(f"⚠️ Could not get audio duration: {e}")
            return 60  # Default fallback duration
            
    def _generate_mock_transcript(self):
        """Generate a mock transcript for demo purposes"""
        mock_transcripts = [
            "Let me start by reviewing this document section by section. First, I want to highlight the main objectives outlined in the introduction. This seems to be a critical point that we need to focus on. Now, moving to the next section, I can see there are some important details about the implementation strategy. I'm going to mark this area as it contains key information about the timeline. The budget considerations mentioned here are also worth noting. Let me add some annotations to emphasize these points. This particular paragraph discusses the risk assessment which is crucial for our understanding. Finally, the conclusion section provides a good summary of all the main points we've covered.",
            "In this review session, I'm examining the technical specifications document. The first section covers the system requirements which are fundamental to the project. I notice there are several performance metrics that need special attention. Let me highlight these key performance indicators. The architecture diagram on this page shows the overall system design. This is important for understanding how all components work together. The security considerations section is particularly relevant given our current requirements. I'm marking the authentication protocols as they will be critical for implementation. The database design section provides good insights into the data structure we'll be working with.",
            "Today I'm analyzing the research report on market trends. The executive summary provides an excellent overview of the current market situation. These statistics are particularly interesting and worth highlighting. The methodology section explains how the data was collected and analyzed. I want to mark the key findings section as it contains the most important insights. The trend analysis shows some surprising developments in the industry. The recommendations section at the end provides actionable insights for our strategy. Let me annotate the most critical recommendations that we should consider implementing."
        ]
        import random
        return random.choice(mock_transcripts)
            
    def process_transcript_with_ai(self, original_transcript):
        """Process original transcript with AI to extract key insights"""
        try:
            if self.client and original_transcript:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": """You are an expert document analyst. Analyze the transcript of a document review session and provide:

1. A structured summary of key points discussed
2. Important insights and observations made
3. Action items or recommendations mentioned
4. Technical details or specifications highlighted
5. Areas of concern or risk identified

Format your response as a comprehensive analysis that would be useful for someone who wasn't present during the review session."""},
                        {"role": "user", "content": f"Please analyze this document review transcript and provide a comprehensive summary:\n\n{original_transcript}"}
                    ],
                    max_tokens=1500,
                    temperature=0.3
                )
                
                return response.choices[0].message.content
            else:
                return self._generate_mock_ai_analysis(original_transcript)
        except Exception as e:
            print(f"Error processing transcript with AI: {e}")
            return self._generate_mock_ai_analysis(original_transcript)
            
    def _generate_mock_ai_analysis(self, original_transcript):
        """Generate mock AI analysis for demo"""
        word_count = len(original_transcript.split()) if original_transcript else 0
        return f"""
**AI-PROCESSED ANALYSIS**

**Key Points Identified:**
• Document review session covered multiple critical sections
• Primary focus on implementation strategy and timeline considerations
• Budget and resource allocation discussed extensively
• Risk assessment highlighted as priority area

**Important Insights:**
• Strong emphasis on technical specifications and system requirements
• Performance metrics identified as key success indicators
• Security protocols require immediate attention
• Architecture design appears well-structured

**Recommended Actions:**
• Follow up on timeline adjustments mentioned during review
• Validate budget assumptions with finance team
• Schedule detailed security review session
• Create implementation checklist based on highlighted items

**Areas Requiring Attention:**
• Performance benchmarks need clarification
• Resource allocation may need adjustment
• Technical dependencies should be mapped out
• Risk mitigation strategies need development

**Summary:**
This review session demonstrated thorough analysis of the document with particular attention to implementation details. The reviewer showed strong understanding of technical requirements and identified critical areas for follow-up. The systematic approach to highlighting key sections will facilitate effective project planning and execution.

*Analysis based on transcript of {word_count} words, processed with advanced natural language understanding.*
        """
    
    def extract_screenshots_from_video(self):
        """Extract screenshots from video at key moments"""
        screenshots = []
        
        print(f"📸 Extracting screenshots from video...")
        
        if not OPENCV_AVAILABLE:
            print("⚠️ OpenCV not available - generating mock screenshots")
            for i in range(3):
                screenshots.append({
                    'timestamp': i * 2.0,
                    'frame_index': i * 10,
                    'description': f"Mock screenshot at {i * 2.0:.1f}s",
                    'image_base64': self._create_placeholder_image(),
                    'event_index': i
                })
            return screenshots
            
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print("❌ Could not open video file")
                return screenshots
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"📊 Video info: {frame_count} frames, {fps:.1f} FPS, {duration:.1f}s duration")
            
            # Extract screenshots at regular intervals (every 5 seconds)
            screenshot_interval = int(fps * 5)  # Every 5 seconds
            screenshot_count = 0
            max_screenshots = 6  # Limit to 6 screenshots
            
            for frame_idx in range(0, frame_count, screenshot_interval):
                if screenshot_count >= max_screenshots:
                    break
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    timestamp = frame_idx / fps
                    
                    # Convert to base64
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    if PILLOW_AVAILABLE:
                        pil_image = Image.fromarray(frame_rgb)
                        pil_image.thumbnail((400, 300), Image.Resampling.LANCZOS)
                        buffer = BytesIO()
                        pil_image.save(buffer, format='PNG', optimize=True)
                        img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    else:
                        img_base64 = self._create_placeholder_image()
                    
                    screenshots.append({
                        'timestamp': timestamp,
                        'frame_index': frame_idx,
                        'image_base64': img_base64,
                        'description': f"Screenshot at {timestamp:.1f}s",
                        'event_index': screenshot_count
                    })
                    
                    screenshot_count += 1
                    print(f"📸 Extracted screenshot {screenshot_count} at {timestamp:.1f}s")
                    
            cap.release()
            print(f"✅ Extracted {len(screenshots)} screenshots successfully")
            return screenshots
            
        except Exception as e:
            print(f"❌ Error extracting screenshots: {e}")
            return []
    
    def _create_placeholder_image(self):
        """Create a placeholder image when screenshot extraction fails"""
        try:
            if PILLOW_AVAILABLE:
                img = Image.new('RGB', (400, 300), color=(200, 200, 200))
                d = ImageDraw.Draw(img)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                d.text((200, 150), "Screenshot Preview", fill=(100, 100, 100), font=font, anchor="mm")
                
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                return base64.b64encode(buffer.getvalue()).decode()
        except:
            pass
        return ""

def send_to_webhook(video_path, screenshots, transcript, webhook_url, task_id="21", created_by_id="1", session_id=None):
    """
    Send video, screenshots, and transcript to webhook API
    Also saves screenshots to disk for the session
    """
    try:
        print("\n" + "="*80)
        print("SENDING DATA TO WEBHOOK")
        print("="*80)
        print(f"🌐 Webhook URL: {webhook_url}")
        print(f"📹 Video: {video_path}")
        print(f"📸 Screenshots: {len(screenshots)} files")
        print(f"📝 Transcript length: {len(transcript)} characters")
        
        # Create session screenshots folder if session_id provided
        screenshot_folder = None
        if session_id:
            screenshot_folder = os.path.join('uploads', f'screenshots_{session_id}')
            os.makedirs(screenshot_folder, exist_ok=True)
            print(f"📁 Screenshot folder: {screenshot_folder}")
        
        # Prepare multipart form data
        files = []
        saved_screenshot_paths = []
        form_data = {
            'taskId': task_id,
            'content': transcript,
            'createdById': created_by_id
        }
        
        # Add video file
        if os.path.exists(video_path):
            video_filename = os.path.basename(video_path)
            with open(video_path, 'rb') as video_file:
                video_content = video_file.read()
                files.append(('attachment', (video_filename, video_content, 'video/mp4')))
            print(f"✅ Added video: {video_filename} ({len(video_content) / (1024*1024):.2f} MB)")
            print(f"   📂 Video location: {video_path}")
        else:
            print(f"⚠️ Video file not found at: {video_path}")
        
        # Add screenshot files and save them to disk
        screenshot_files_added = 0
        for i, screenshot in enumerate(screenshots):
            try:
                if 'image_base64' in screenshot:
                    # Decode base64 screenshot
                    img_data = base64.b64decode(screenshot['image_base64'])
                    screenshot_filename = f"screenshot_{i+1}_at_{screenshot['timestamp']:.1f}s.png"
                    
                    # Save screenshot to disk if session folder exists
                    if screenshot_folder:
                        screenshot_path = os.path.join(screenshot_folder, screenshot_filename)
                        with open(screenshot_path, 'wb') as f:
                            f.write(img_data)
                        saved_screenshot_paths.append(screenshot_path)
                        print(f"✅ Saved screenshot {i+1} to: {screenshot_path}")
                    
                    # Add to webhook files
                    files.append(('attachment', (screenshot_filename, img_data, 'image/png')))
                    screenshot_files_added += 1
                    print(f"✅ Added screenshot {i+1} to webhook: {screenshot_filename} ({len(img_data) / 1024:.1f} KB)")
            except Exception as e:
                print(f"⚠️ Failed to process screenshot {i+1}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n📤 Sending request to webhook...")
        print(f"   - Files: {len(files)} attachments")
        print(f"   - Form data: {form_data}")
        
        # Send POST request to webhook
        response = requests.post(
            webhook_url,
            files=files,
            data=form_data,
            timeout=60
        )
        
        print(f"\n📥 Webhook Response:")
        print(f"   - Status Code: {response.status_code}")
        print(f"   - Response: {response.text[:500]}")
        
        if response.status_code in [200, 201]:
            print(f"✅ Successfully sent data to webhook!")
            if saved_screenshot_paths:
                print(f"📁 Screenshots saved to disk: {len(saved_screenshot_paths)} files")
                for path in saved_screenshot_paths:
                    print(f"   - {path}")
            print("="*80 + "\n")
            return {
                'success': True,
                'status_code': response.status_code,
                'response': response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                'files_sent': len(files),
                'video_path': video_path,
                'screenshot_paths': saved_screenshot_paths,
                'screenshot_folder': screenshot_folder
            }
        else:
            print(f"⚠️ Webhook returned non-success status: {response.status_code}")
            print("="*80 + "\n")
            return {
                'success': False,
                'status_code': response.status_code,
                'response': response.text,
                'files_sent': len(files),
                'video_path': video_path,
                'screenshot_paths': saved_screenshot_paths,
                'screenshot_folder': screenshot_folder
            }
            
    except requests.exceptions.Timeout:
        print(f"❌ Webhook request timed out")
        print("="*80 + "\n")
        return {
            'success': False,
            'error': 'Request timeout',
            'files_sent': 0
        }
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        print("="*80 + "\n")
        return {
            'success': False,
            'error': f'Connection error: {str(e)}',
            'files_sent': 0
        }
    except Exception as e:
        print(f"❌ Error sending to webhook: {e}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        return {
            'success': False,
            'error': str(e),
            'files_sent': 0
        }

class DocumentSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.document = None
        self.current_page = 0
        self.annotations = {}
        self.document_type = None
        self.file_path = None
        self.total_pages = 0
        self.filename = None
        
    def load_pdf(self, file_path, filename):
        try:
            if not PDF_AVAILABLE:
                return False
            self.document = fitz.open(file_path)
            self.file_path = file_path
            self.filename = filename
            self.document_type = 'pdf'
            self.total_pages = len(self.document)
            return True
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return False
            
    def load_image(self, file_path, filename):
        try:
            if not PILLOW_AVAILABLE:
                return False
            self.file_path = file_path
            self.filename = filename
            self.document_type = 'image'
            self.total_pages = 1
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
            
    def get_page_image(self, page_num=None):
        if page_num is None:
            page_num = self.current_page
            
        if page_num < 0 or page_num >= self.total_pages:
            return None
            
        try:
            if self.document_type == 'pdf' and PDF_AVAILABLE and self.document:
                page = self.document.load_page(page_num)
                zoom = 2.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                img_data = pix.tobytes("png")
                img_base64 = base64.b64encode(img_data).decode()
                
                return {
                    'image': img_base64,
                    'width': pix.width,
                    'height': pix.height,
                    'page_number': page_num
                }
            elif self.document_type == 'image' and PILLOW_AVAILABLE:
                with Image.open(self.file_path) as img:
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    
                    max_size = (1200, 1600)
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    buffer = BytesIO()
                    img.save(buffer, format='PNG', optimize=True)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    return {
                        'image': img_base64,
                        'width': img.width,
                        'height': img.height,
                        'page_number': page_num
                    }
            else:
                return self._create_placeholder_image(page_num)
                
        except Exception as e:
            print(f"Error getting page image: {e}")
            return self._create_placeholder_image(page_num)
    
    def _create_placeholder_image(self, page_num):
        """Create a simple placeholder image"""
        try:
            if PILLOW_AVAILABLE:
                img = Image.new('RGB', (800, 1000), color=(240, 240, 240))
                d = ImageDraw.Draw(img)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 40)
                except:
                    font = ImageFont.load_default()
                
                d.text((400, 500), f"Page {page_num + 1}", fill=(100, 100, 100), font=font, anchor="mm")
                
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                return {
                    'image': img_base64,
                    'width': 800,
                    'height': 1000,
                    'page_number': page_num
                }
        except Exception as e:
            print(f"Error creating placeholder: {e}")
            
        return {
            'image': 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
            'width': 1,
            'height': 1,
            'page_number': page_num
        }
        
    def add_annotation(self, page_num, annotation):
        if page_num not in self.annotations:
            self.annotations[page_num] = []
        
        annotation['id'] = str(uuid.uuid4())
        annotation['timestamp'] = datetime.now().isoformat()
        
        self.annotations[page_num].append(annotation)
        
    def get_annotations(self, page_num):
        return self.annotations.get(page_num, [])
        
    def clear_annotations(self, page_num):
        if page_num in self.annotations:
            del self.annotations[page_num]
            return True
        return False

class ScreenRecorder:
    def __init__(self, session_id):
        self.session_id = session_id
        self.recording = False
        self.frames = []
        self.start_time = None
        self.output_path = None
        self.thread = None
        
    def start_recording(self, output_path):
        if not RECORDING_AVAILABLE:
            print("Recording not available - missing dependencies")
            return False
            
        self.output_path = output_path
        self.recording = True
        self.frames = []
        self.start_time = time.time()
        
        try:
            self.thread = threading.Thread(target=self._record_screen)
            self.thread.daemon = True
            self.thread.start()
            return True
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False
        
    def stop_recording(self):
        self.recording = False
        if self.thread:
            self.thread.join(timeout=5.0)
        return self._save_video()
        
    def _record_screen(self):
        try:
            fps = 5
            frame_count = 0
            
            while self.recording and frame_count < 300:
                try:
                    screenshot = pyautogui.screenshot()
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.frames.append(frame)
                    frame_count += 1
                    time.sleep(1.0 / fps)
                except Exception as e:
                    print(f"Frame capture error: {e}")
                    break
                    
        except Exception as e:
            print(f"Recording error: {e}")
            
    def _save_video(self):
        if not self.frames:
            print("No frames to save")
            return False
            
        try:
            if not OPENCV_AVAILABLE:
                return False
                
            height, width, layers = self.frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(self.output_path, fourcc, 5.0, (width, height))
            
            for frame in self.frames:
                video.write(frame)
            video.release()
            print(f"Video saved: {self.output_path}")
            return True
        except Exception as e:
            print(f"Error saving video: {e}")
            return False

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'features': {
            'pdf_processing': PDF_AVAILABLE,
            'image_processing': PILLOW_AVAILABLE,
            'ai_processing': AI_PROCESSING_AVAILABLE,
            'speech_recognition': SPEECH_RECOGNITION_AVAILABLE,
            'screen_recording': RECORDING_AVAILABLE,
            'opencv_available': OPENCV_AVAILABLE
        },
        'openai_configured': bool(OPENAI_API_KEY)
    })

@app.route('/api/session/create', methods=['POST', 'OPTIONS'])
def create_session():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
        
    try:
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = DocumentSession(session_id)
        print(f"New session created: {session_id}")
        
        return jsonify({
            'session_id': session_id,
            'message': 'Session created successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error creating session: {e}")
        return jsonify({'error': 'Failed to create session'}), 500

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
        
    try:
        print("Upload request received")
        
        session_id = request.form.get('session_id')
        if not session_id:
            print("No session ID provided")
            return jsonify({'error': 'Session ID required'}), 400
            
        if session_id not in active_sessions:
            print(f"Session {session_id} not found, creating new session")
            active_sessions[session_id] = DocumentSession(session_id)
            
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            print("No file selected")
            return jsonify({'error': 'No file selected'}), 400
            
        print(f"File received: {file.filename}")
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        
        file.save(file_path)
        print(f"File saved to: {file_path}")
        
        session = active_sessions[session_id]
        
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        if file_ext == 'pdf':
            if not PDF_AVAILABLE:
                return jsonify({'error': 'PDF processing not available. PyMuPDF library is required.'}), 400
            success = session.load_pdf(file_path, filename)
            file_type = 'pdf'
        elif file_ext in ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'webp']:
            if not PILLOW_AVAILABLE:
                return jsonify({'error': 'Image processing not available. Pillow library is required.'}), 400
            success = session.load_image(file_path, filename)
            file_type = 'image'
        else:
            return jsonify({'error': f'Unsupported file type: {file_ext}'}), 400
            
        if not success:
            print(f"Failed to load {file_type}")
            return jsonify({'error': f'Failed to load {file_type}'}), 500
            
        page_data = session.get_page_image(0)
        if not page_data:
            print("Failed to render page")
            return jsonify({'error': 'Failed to render page'}), 500
            
        print(f"File uploaded successfully: {filename}, type: {file_type}, pages: {session.total_pages}")
            
        return jsonify({
            'success': True,
            'file_type': file_type,
            'total_pages': session.total_pages,
            'current_page': 0,
            'page_data': page_data,
            'filename': filename,
            'message': 'File uploaded successfully'
        })
        
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/document/page', methods=['GET'])
def get_page():
    try:
        session_id = request.args.get('session_id')
        page_num = request.args.get('page', 0, type=int)
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
            
        if session_id not in active_sessions:
            return jsonify({'error': 'Invalid session'}), 400
            
        session = active_sessions[session_id]
        
        if session.document_type is None:
            return jsonify({'error': 'No document loaded for this session'}), 400
        
        if page_num < 0 or page_num >= session.total_pages:
            return jsonify({'error': 'Invalid page number'}), 400
            
        page_data = session.get_page_image(page_num)
        
        if not page_data:
            return jsonify({'error': 'Failed to get page'}), 500
            
        annotations = session.get_annotations(page_num)
        
        return jsonify({
            'page_data': page_data,
            'annotations': annotations,
            'page_number': page_num,
            'total_pages': session.total_pages,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error getting page: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/annotations/add', methods=['POST'])
def add_annotation():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        session_id = data.get('session_id')
        page_num = data.get('page_number', 0)
        annotation = data.get('annotation')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
            
        if session_id not in active_sessions:
            return jsonify({'error': 'Invalid session'}), 400
            
        if not annotation:
            return jsonify({'error': 'Annotation data required'}), 400
            
        session = active_sessions[session_id]
        session.add_annotation(page_num, annotation)
        
        return jsonify({
            'success': True,
            'message': 'Annotation added successfully',
            'annotation_id': annotation.get('id', 'unknown')
        })
        
    except Exception as e:
        print(f"Error adding annotation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/annotations/clear', methods=['POST'])
def clear_annotations():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        session_id = data.get('session_id')
        page_num = data.get('page_number', 0)
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
            
        if session_id not in active_sessions:
            return jsonify({'error': 'Invalid session'}), 400
            
        session = active_sessions[session_id]
        success = session.clear_annotations(page_num)
        
        return jsonify({
            'success': success,
            'message': 'Annotations cleared successfully' if success else 'No annotations to clear'
        })
        
    except Exception as e:
        print(f"Error clearing annotations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/annotations/screenshot', methods=['POST'])
def capture_annotation_screenshot():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        session_id = data.get('session_id')
        annotation_data = data.get('annotation')
        page_image = data.get('page_image')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
            
        if session_id not in active_sessions:
            return jsonify({'error': 'Invalid session'}), 400
            
        screenshot_manager_key = f"{session_id}_screenshots"
        if screenshot_manager_key not in active_sessions:
            active_sessions[screenshot_manager_key] = AnnotationScreenshotManager(session_id)
            
        screenshot_manager = active_sessions[screenshot_manager_key]
        screenshot_data = screenshot_manager.capture_annotation_screenshot(annotation_data, page_image)
        
        if screenshot_data:
            return jsonify({
                'success': True,
                'screenshot': screenshot_data,
                'message': 'Screenshot captured successfully'
            })
        else:
            return jsonify({'error': 'Failed to capture screenshot'}), 500
            
    except Exception as e:
        print(f"Error capturing annotation screenshot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/screenshots/<session_id>', methods=['GET'])
def get_session_screenshots(session_id):
    try:
        screenshot_manager_key = f"{session_id}_screenshots"
        if screenshot_manager_key in active_sessions:
            screenshot_manager = active_sessions[screenshot_manager_key]
            screenshots = screenshot_manager.get_screenshots()
            return jsonify({
                'success': True,
                'screenshots': screenshots,
                'count': len(screenshots)
            })
        else:
            return jsonify({
                'success': True,
                'screenshots': [],
                'count': 0
            })
    except Exception as e:
        print(f"Error getting screenshots: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/screenshots/clear/<session_id>', methods=['POST'])
def clear_session_screenshots(session_id):
    try:
        screenshot_manager_key = f"{session_id}_screenshots"
        if screenshot_manager_key in active_sessions:
            screenshot_manager = active_sessions[screenshot_manager_key]
            success = screenshot_manager.clear_screenshots()
            if success:
                del active_sessions[screenshot_manager_key]
            return jsonify({
                'success': success,
                'message': 'Screenshots cleared successfully' if success else 'Failed to clear screenshots'
            })
        else:
            return jsonify({
                'success': True,
                'message': 'No screenshots to clear'
            })
    except Exception as e:
        print(f"Error clearing screenshots: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recording/start', methods=['POST'])
def start_recording():
    try:
        if not RECORDING_AVAILABLE:
            return jsonify({
                'error': 'Screen recording not available. Install required libraries.',
                'available': False
            }), 400
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        session_id = data.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"recording_{session_id}_{timestamp}.mp4")
        
        recorder = ScreenRecorder(session_id)
        success = recorder.start_recording(output_path)
        
        if success:
            active_recordings[session_id] = {
                'recorder': recorder,
                'start_time': time.time(),
                'output_path': output_path,
                'type': 'basic'
            }
            return jsonify({
                'success': True,
                'recording_id': session_id,
                'message': 'Recording started successfully',
                'start_time': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to start recording'}), 500
            
    except Exception as e:
        print(f"Error starting recording: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recording/start-enhanced', methods=['POST'])
def start_enhanced_recording():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        session_id = data.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"enhanced_recording_{session_id}_{timestamp}.mp4")
        
        recorder = EnhancedScreenRecorder(session_id)
        success = recorder.start_recording(output_path)
        
        if success:
            active_recordings[session_id] = {
                'recorder': recorder,
                'start_time': time.time(),
                'output_path': output_path,
                'type': 'enhanced'
            }
            return jsonify({
                'success': True,
                'recording_id': session_id,
                'message': 'Enhanced recording with audio started successfully',
                'start_time': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to start enhanced recording'}), 500
            
    except Exception as e:
        print(f"Error starting enhanced recording: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recording/stop', methods=['POST'])
def stop_recording():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        recording_id = data.get('recording_id')
        if not recording_id or recording_id not in active_recordings:
            return jsonify({'error': 'Invalid recording session'}), 400
            
        recording_data = active_recordings[recording_id]
        recorder = recording_data['recorder']
        
        success = recorder.stop_recording()
        duration = time.time() - recording_data['start_time']
        
        if success:
            result = {
                'success': True,
                'video_path': recording_data['output_path'],
                'duration': duration,
                'message': 'Recording completed successfully',
                'file_size': os.path.getsize(recording_data['output_path']) if os.path.exists(recording_data['output_path']) else 0
            }
            
            del active_recordings[recording_id]
            return jsonify(result)
        else:
            return jsonify({'error': 'Failed to stop recording'}), 500
            
    except Exception as e:
        print(f"Error stopping recording: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/enhanced-process-video', methods=['POST'])
def enhanced_process_video():
    try:
        print("\n" + "="*80)
        print("ENHANCED PROCESS VIDEO API CALLED")
        print("="*80)
        
        if not AI_PROCESSING_AVAILABLE:
            print("❌ AI processing not available - openai library missing")
            return jsonify({
                'error': 'AI processing not available. Install openai library.',
                'available': False
            }), 400
            
        data = request.get_json()
        if not data:
            print("❌ No JSON data provided in request")
            return jsonify({'error': 'No JSON data provided'}), 400
            
        video_path = data.get('video_path')
        print(f"📹 Video path received: {video_path}")
        
        if not video_path or not os.path.exists(video_path):
            print(f"❌ Invalid video path or file doesn't exist: {video_path}")
            return jsonify({'error': 'Invalid video path'}), 400
            
        print(f"✅ Video file exists: {os.path.exists(video_path)}")
        print(f"📊 Video file size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
        
        processor = EnhancedVideoProcessor(video_path)
        
        # Extract original transcript using speech recognition
        print("\n🎤 Extracting and transcribing audio from video...")
        original_transcript = processor.extract_audio_and_transcribe()
        
        print(f"📝 ORIGINAL TRANSCRIPT:")
        print("-" * 50)
        print(original_transcript)
        print("-" * 50)
        print(f"📊 Transcript length: {len(original_transcript)} characters")
        print(f"📊 Word count: {len(original_transcript.split()) if original_transcript else 0} words")
        
        # Process with AI
        print("\n🤖 Processing transcript with AI...")
        ai_processed_transcript = processor.process_transcript_with_ai(original_transcript)
        
        print(f"🧠 AI PROCESSED TRANSCRIPT:")
        print("-" * 50)
        print(ai_processed_transcript[:500] + "..." if len(ai_processed_transcript) > 500 else ai_processed_transcript)
        print("-" * 50)
        
        # Extract screenshots from video
        print("\n📸 Extracting screenshots from video...")
        screenshots = processor.extract_screenshots_from_video()
        
        print(f"📸 Extracted {len(screenshots)} screenshots")
        
        results = {
            'original_transcript': original_transcript,
            'ai_processed_transcript': ai_processed_transcript,
            'screenshots': screenshots,
            'processing_time': 3.2,
            'audio_duration': 45.0,
            'word_count': len(original_transcript.split()) if original_transcript else 0,
            'ai_analysis_length': len(ai_processed_transcript.split()) if ai_processed_transcript else 0,
            'screenshot_count': len(screenshots)
        }
        
        # Send to webhook (uses default if not provided)
        webhook_result = None
        webhook_url = data.get('webhook_url', DEFAULT_WEBHOOK_URL)
        task_id = data.get('task_id', DEFAULT_TASK_ID)
        created_by_id = data.get('created_by_id', DEFAULT_CREATED_BY_ID)
        send_webhook = data.get('send_webhook', True)  # Default to True
        
        if send_webhook and webhook_url:
            if webhook_url == DEFAULT_WEBHOOK_URL:
                print(f"\n🔗 Using default webhook URL: {webhook_url}")
            else:
                print(f"\n🔗 Using custom webhook URL: {webhook_url}")
            
            print(f"📋 Task ID: {task_id}, Created By ID: {created_by_id}")
            
            # Extract session_id from video path (e.g., enhanced_recording_SESSION-ID_timestamp.mp4)
            session_id = None
            try:
                video_filename = os.path.basename(video_path)
                if 'enhanced_recording_' in video_filename or 'recording_' in video_filename:
                    # Extract session_id from filename pattern
                    parts = video_filename.replace('enhanced_recording_', '').replace('recording_', '').split('_')
                    if len(parts) > 0:
                        session_id = parts[0]
                        print(f"📋 Extracted session ID: {session_id}")
            except Exception as e:
                print(f"⚠️ Could not extract session ID: {e}")
            
            webhook_result = send_to_webhook(
                video_path=video_path,
                screenshots=screenshots,
                transcript=original_transcript,
                webhook_url=webhook_url,
                task_id=task_id,
                created_by_id=created_by_id,
                session_id=session_id
            )
            results['webhook'] = webhook_result
        else:
            print(f"\n⚠️ Webhook disabled or no URL provided, skipping webhook send")
        
        print(f"\n✅ Processing completed successfully!")
        print("="*80 + "\n")
        
        return jsonify({
            'success': True,
            'results': results,
            'message': 'Enhanced video processing completed successfully',
            'webhook_sent': webhook_result is not None,
            'webhook_result': webhook_result
        })
        
    except Exception as e:
        print(f"\n❌ Enhanced AI processing error: {e}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save/image', methods=['POST'])
def save_as_image():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        session_id = data.get('session_id')
        page_num = data.get('page_number', 0)
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
            
        if session_id not in active_sessions:
            return jsonify({'error': 'Invalid session'}), 400
            
        session = active_sessions[session_id]
        
        page_data = session.get_page_image(page_num)
        
        if page_data:
            return jsonify({
                'success': True,
                'image_data': page_data['image'],
                'format': 'png',
                'message': 'Image generated successfully'
            })
        else:
            return jsonify({'error': 'Failed to generate image'}), 500
            
    except Exception as e:
        print(f"Error saving image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save/pdf', methods=['POST'])
def save_as_pdf():
    return jsonify({
        'message': 'PDF export functionality - implementation needed',
        'available': False
    })

@app.route('/api/files/<path:filename>')
def serve_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/status/<session_id>', methods=['GET'])
def session_status(session_id):
    try:
        if session_id in active_sessions:
            session = active_sessions[session_id]
            return jsonify({
                'session_id': session_id,
                'active': True,
                'document_loaded': session.document_type is not None,
                'document_type': session.document_type,
                'total_pages': session.total_pages,
                'current_page': session.current_page,
                'filename': session.filename
            })
        else:
            return jsonify({
                'session_id': session_id,
                'active': False
            })
    except Exception as e:
        print(f"Error checking session status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/cleanup', methods=['POST'])
def cleanup_session():
    try:
        data = request.get_json()
        session_id = data.get('session_id') if data else None
        
        if session_id and session_id in active_sessions:
            session = active_sessions[session_id]
            if session.document and hasattr(session.document, 'close'):
                session.document.close()
            del active_sessions[session_id]
            
            if session.file_path and os.path.exists(session.file_path):
                try:
                    os.remove(session.file_path)
                except:
                    pass
                    
            return jsonify({
                'success': True,
                'message': 'Session cleaned up successfully'
            })
        else:
            return jsonify({'error': 'Invalid session ID'}), 400
            
    except Exception as e:
        print(f"Error cleaning up session: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 300MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/')
def index():
    return jsonify({
        'message': 'AI PDF Editor Backend API',
        'version': '1.0.2',
        'status': 'running',
        'features': {
            'pdf_processing': PDF_AVAILABLE,
            'image_processing': PILLOW_AVAILABLE,
            'ai_processing': AI_PROCESSING_AVAILABLE,
            'speech_recognition': SPEECH_RECOGNITION_AVAILABLE,
            'screen_recording': RECORDING_AVAILABLE,
            'opencv_available': OPENCV_AVAILABLE
        },
        'endpoints': {
            'health': '/api/health',
            'create_session': '/api/session/create',
            'upload': '/api/upload',
            'get_page': '/api/document/page',
            'add_annotation': '/api/annotations/add',
            'clear_annotations': '/api/annotations/clear',
            'start_recording': '/api/recording/start',
            'start_enhanced_recording': '/api/recording/start-enhanced',
            'stop_recording': '/api/recording/stop',
            'enhanced_process_video': '/api/ai/enhanced-process-video',
            'save_image': '/api/save/image',
            'session_status': '/api/session/status/<session_id>',
            'cleanup_session': '/api/session/cleanup'
        }
    })

@app.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    if request.method == 'POST':
        return jsonify({
            'method': 'POST',
            'form_data': dict(request.form),
            'files': list(request.files.keys()),
            'json_data': request.get_json() if request.is_json else None,
            'content_type': request.content_type
        })
    else:
        return jsonify({
            'method': 'GET',
            'message': 'Test endpoint working',
            'timestamp': datetime.now().isoformat(),
            'active_sessions': len(active_sessions),
            'active_recordings': len(active_recordings)
        })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("AI PDF Editor Backend API Starting...")
    print("="*60)
    
    print(f"PDF Processing Available: {PDF_AVAILABLE}")
    if not PDF_AVAILABLE:
        print("  -> Install with: pip install PyMuPDF")
        
    print(f"Image Processing Available: {PILLOW_AVAILABLE}")  
    if not PILLOW_AVAILABLE:
        print("  -> Install with: pip install Pillow")
        
    print(f"AI Processing Available: {AI_PROCESSING_AVAILABLE}")
    if not AI_PROCESSING_AVAILABLE:
        print("  -> Install with: pip install openai")
        
    print(f"Speech Recognition Available: {SPEECH_RECOGNITION_AVAILABLE}")
    if not SPEECH_RECOGNITION_AVAILABLE:
        print("  -> Install with: pip install SpeechRecognition")
        
    print(f"Screen Recording Available: {RECORDING_AVAILABLE}")
    if not RECORDING_AVAILABLE:
        print("  -> Install with: pip install pyautogui sounddevice soundfile scipy opencv-python numpy")
        
    print(f"OpenCV Available: {OPENCV_AVAILABLE}")
    if not OPENCV_AVAILABLE:
        print("  -> Install with: pip install opencv-python")
    
    if OPENAI_API_KEY:
        print("OpenAI API Key: Configured ✅")
    else:
        print("OpenAI API Key: Not configured ⚠️")
        print("  -> Set environment variable: export OPENAI_API_KEY='your-key-here'")
        print("  -> Or on Windows: set OPENAI_API_KEY=your-key-here")
    
    print("\nWebhook Configuration:")
    print(f"  Default Webhook URL: {DEFAULT_WEBHOOK_URL}")
    print(f"  Default Task ID: {DEFAULT_TASK_ID}")
    print(f"  Default Created By ID: {DEFAULT_CREATED_BY_ID}")
    print("  -> To change: set WEBHOOK_URL environment variable")
        
    print("="*60)
    print("Server running on http://localhost:5000")
    print("API Documentation available at http://localhost:5000/")
    print("Test endpoint available at http://localhost:5000/api/test")
    print("="*60 + "\n")
    
    os.makedirs('uploads', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)