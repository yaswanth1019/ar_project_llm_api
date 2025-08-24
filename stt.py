import whisper
import io
import os
import wave
import logging
import time
from pathlib import Path
import hashlib
import uuid
import gc
import tempfile
import shutil
import numpy as np

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

# Local temp directory in current folder
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

def initialize_whisper_model(model_size="base.en"):
    """Initialize Whisper model with enhanced error handling"""
    global model
    if model is None:
        try:
            logger.info(f"Loading Whisper model: {model_size}")
            model = whisper.load_model(model_size)
            logger.info("✓ Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            # Try fallback to smaller model
            try:
                logger.info("Trying fallback to tiny.en model...")
                model = whisper.load_model("tiny.en")
                logger.info("✓ Fallback Whisper model loaded successfully")
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
                raise Exception(f"Could not load any Whisper model: {e}")
    return model

def validate_audio_data(audio_bytes: bytes) -> bool:
    """Validate audio data before processing"""
    if not audio_bytes or len(audio_bytes) == 0:
        raise ValueError("Empty audio data provided")
    
    if len(audio_bytes) < 1000:  # Very small file, likely corrupted
        raise ValueError("Audio data too small, possibly corrupted")
    
    logger.info(f"Audio data size: {len(audio_bytes)} bytes")
    return True

def create_temp_audio_file(audio_bytes: bytes) -> str:
    """Create a temporary audio file using system temp directory as fallback"""
    try:
        # First try local temp directory
        try:
            TEMP_DIR.mkdir(exist_ok=True)
            
            # Create unique filename
            unique_id = str(uuid.uuid4())[:8]
            timestamp = int(time.time())
            filename = f"audio_{unique_id}_{timestamp}.wav"
            temp_path = TEMP_DIR / filename
            
            # Write audio data to file with proper closing
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)
                f.flush()
                os.fsync(f.fileno())
            
            # Verify file was created correctly
            if not temp_path.exists():
                raise RuntimeError("Failed to create temporary file in local directory")
            
            file_size = temp_path.stat().st_size
            if file_size == 0:
                raise RuntimeError("Created temporary file is empty")
            
            logger.info(f"Created temp file: {temp_path} ({file_size} bytes)")
            return str(temp_path)
            
        except Exception as local_error:
            logger.warning(f"Failed to create file in local temp dir: {local_error}")
            logger.info("Falling back to system temp directory...")
            
            # Fallback to system temp directory
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_path = temp_file.name
            
            # Verify system temp file
            if not os.path.exists(temp_path):
                raise RuntimeError("Failed to create temporary file in system temp")
            
            file_size = os.path.getsize(temp_path)
            if file_size == 0:
                raise RuntimeError("System temp file is empty")
            
            logger.info(f"Created system temp file: {temp_path} ({file_size} bytes)")
            return temp_path
        
    except Exception as e:
        logger.error(f"Failed to create temporary audio file: {e}")
        raise

def safe_remove_file(file_path: str, max_attempts: int = 5):
    """Safely remove file with Windows-compatible retry logic"""
    for attempt in range(max_attempts):
        try:
            if os.path.exists(file_path):
                # Force garbage collection to release any file handles
                gc.collect()
                time.sleep(0.1)
                
                os.unlink(file_path)
                logger.info(f"Successfully removed temp file: {file_path}")
                return True
        except (PermissionError, OSError) as e:
            if attempt < max_attempts - 1:
                logger.warning(f"Attempt {attempt + 1} to remove {file_path} failed: {e}, retrying...")
                time.sleep(0.5)
            else:
                logger.warning(f"Could not remove {file_path} after {max_attempts} attempts: {e}")
                return False
        except Exception as e:
            logger.warning(f"Unexpected error removing {file_path}: {e}")
            return False
    return False

def cleanup_old_temp_files(max_age_minutes: int = 60):
    """Clean up old temporary files"""
    try:
        if not TEMP_DIR.exists():
            return
        
        current_time = time.time()
        cleaned_count = 0
        
        for file_path in TEMP_DIR.glob("audio_*.wav"):
            try:
                file_age = current_time - file_path.stat().st_mtime
                if file_age > (max_age_minutes * 60):
                    if safe_remove_file(str(file_path)):
                        cleaned_count += 1
            except Exception as e:
                logger.warning(f"Could not clean up old file {file_path}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old temporary files")
            
    except Exception as e:
        logger.warning(f"Error during temp file cleanup: {e}")

def transcribe_with_numpy_array(audio_bytes: bytes) -> str:
    """Alternative transcription method using numpy array instead of file"""
    try:
        logger.info("Attempting transcription with numpy array method...")
        
        # Initialize model
        whisper_model = initialize_whisper_model()
        
        # Create temporary file to load audio
        temp_path = None
        try:
            temp_path = create_temp_audio_file(audio_bytes)
            
            # Load audio into numpy array using whisper's loader
            audio_array = whisper.load_audio(temp_path)
            
            # Remove temp file immediately after loading
            if temp_path:
                safe_remove_file(temp_path)
                temp_path = None
            
            # Transcribe using the audio array directly
            result = whisper_model.transcribe(
                audio_array,
                fp16=False,
                verbose=False,
                temperature=0.0
            )
            
            if not result or "text" not in result:
                raise ValueError("Invalid transcription result")
            
            transcription = result["text"].strip()
            logger.info(f"Numpy array transcription successful: '{transcription[:100]}{'...' if len(transcription) > 100 else ''}'")
            
            return transcription
            
        finally:
            if temp_path:
                safe_remove_file(temp_path)
                
    except Exception as e:
        logger.error(f"Numpy array transcription failed: {e}")
        raise

def transcribe_with_copy_file(audio_bytes: bytes) -> str:
    """Alternative method that copies file to a more stable location"""
    try:
        logger.info("Attempting transcription with file copy method...")
        
        whisper_model = initialize_whisper_model()
        
        # Create file in system temp with a more permanent approach
        temp_dir = tempfile.mkdtemp()
        temp_filename = f"whisper_audio_{int(time.time())}.wav"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        try:
            # Write audio data
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)
                f.flush()
                os.fsync(f.fileno())
            
            # Verify file exists and is readable
            if not os.path.exists(temp_path):
                raise FileNotFoundError(f"Could not create file: {temp_path}")
            
            file_size = os.path.getsize(temp_path)
            logger.info(f"Created stable temp file: {temp_path} ({file_size} bytes)")
            
            # Wait a moment to ensure file is fully written
            time.sleep(0.5)
            
            # Double-check file still exists
            if not os.path.exists(temp_path):
                raise FileNotFoundError(f"File disappeared: {temp_path}")
            
            # Transcribe
            result = whisper_model.transcribe(
                temp_path,
                fp16=False,
                verbose=False,
                temperature=0.0,
                best_of=1
            )
            
            if not result or "text" not in result:
                raise ValueError("Invalid transcription result")
            
            transcription = result["text"].strip()
            logger.info(f"File copy transcription successful: '{transcription[:100]}{'...' if len(transcription) > 100 else ''}'")
            
            return transcription
            
        finally:
            # Clean up the entire temp directory
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up temp directory {temp_dir}: {e}")
                
    except Exception as e:
        logger.error(f"File copy transcription failed: {e}")
        raise

def transcribe(audio_bytes: bytes, max_retries: int = 3) -> str:
    """
    Transcribe audio bytes to text using Whisper with multiple fallback methods
    """
    
    # Clean up old temp files first
    cleanup_old_temp_files()
    
    # Validate input
    validate_audio_data(audio_bytes)
    
    # Try multiple methods in order of preference
    methods = [
        ("numpy array", transcribe_with_numpy_array),
        ("file copy", transcribe_with_copy_file),
        ("original", transcribe_original_method)
    ]
    
    for method_name, method_func in methods:
        try:
            logger.info(f"Trying {method_name} method...")
            result = method_func(audio_bytes)
            if result and len(result.strip()) > 0:
                logger.info(f"Success with {method_name} method!")
                return result
        except Exception as e:
            logger.error(f"{method_name} method failed: {e}")
            continue
    
    # If all methods failed
    raise Exception("All transcription methods failed")

def transcribe_original_method(audio_bytes: bytes, max_retries: int = 3) -> str:
    """Original transcription method with improvements"""
    
    for attempt in range(max_retries):
        temp_path = None
        try:
            logger.info(f"Original method attempt {attempt + 1}/{max_retries}")
            
            # Initialize model
            whisper_model = initialize_whisper_model()
            
            # Create temporary file
            temp_path = create_temp_audio_file(audio_bytes)
            
            # Additional verification
            if not os.path.exists(temp_path):
                raise FileNotFoundError(f"Temporary file not found: {temp_path}")
            
            file_size = os.path.getsize(temp_path)
            if file_size == 0:
                raise ValueError("Temporary file is empty")
            
            logger.info(f"Processing audio file: {temp_path} ({file_size} bytes)")
            
            # Try to validate it's a proper audio file
            try:
                with wave.open(temp_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    duration = frames / sample_rate if sample_rate > 0 else 0
                    logger.info(f"WAV file: {frames} frames, {sample_rate}Hz, {duration:.2f}s")
            except wave.Error:
                logger.info("File is not WAV format, but Whisper should handle it")
            
            # Wait a moment to ensure file stability
            time.sleep(0.2)
            
            # Final existence check
            if not os.path.exists(temp_path):
                raise FileNotFoundError(f"Temp file disappeared: {temp_path}")
            
            logger.info("Starting Whisper transcription...")
            
            # Transcription options
            transcribe_options = {
                "fp16": False,
                "verbose": False,
                "temperature": 0.0,
                "best_of": 1,
                "beam_size": 1 if attempt > 0 else None
            }
            
            # Remove None values
            transcribe_options = {k: v for k, v in transcribe_options.items() if v is not None}
            
            logger.info(f"Calling Whisper transcribe on: {temp_path}")
            logger.info(f"File exists before transcribe: {os.path.exists(temp_path)}")
            
            # The actual transcription call
            result = whisper_model.transcribe(temp_path, **transcribe_options)
            
            logger.info("Whisper transcription completed successfully")
            
            if not result or "text" not in result:
                raise ValueError("Invalid transcription result from Whisper")
            
            transcription = result["text"].strip()
            
            if not transcription:
                raise ValueError("Empty transcription result")
            
            logger.info(f"Transcription successful: '{transcription[:100]}{'...' if len(transcription) > 100 else ''}'")
            
            return transcription
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                logger.info(f"Waiting before retry attempt {attempt + 2}")
                time.sleep(2)
        
        finally:
            if temp_path:
                safe_remove_file(temp_path)
    
    raise Exception(f"All {max_retries} original method attempts failed")

def transcribe_with_options(audio_bytes: bytes, model_size="base.en", language="en") -> str:
    """Transcribe with custom options using the improved methods"""
    try:
        validate_audio_data(audio_bytes)
        
        # Try numpy array method first
        try:
            local_model = whisper.load_model(model_size)
            temp_path = create_temp_audio_file(audio_bytes)
            
            try:
                audio_array = whisper.load_audio(temp_path)
                safe_remove_file(temp_path)
                
                result = local_model.transcribe(
                    audio_array,
                    language=language,
                    task="transcribe",
                    fp16=False,
                    verbose=False
                )
                
                return result["text"].strip()
                
            finally:
                if os.path.exists(temp_path):
                    safe_remove_file(temp_path)
                    
        except Exception as e:
            logger.warning(f"Numpy method failed, trying file copy method: {e}")
            
            # Fallback to file copy method
            return transcribe_with_copy_file(audio_bytes)
            
    except Exception as e:
        logger.error(f"Custom transcription error: {e}")
        raise

def transcribe_with_language_detection(audio_bytes: bytes) -> dict:
    """Transcribe audio with language detection using improved methods"""
    try:
        validate_audio_data(audio_bytes)
        whisper_model = initialize_whisper_model()
        
        # Use numpy array method for language detection
        temp_path = create_temp_audio_file(audio_bytes)
        
        try:
            # Load audio array
            audio_array = whisper.load_audio(temp_path)
            safe_remove_file(temp_path)
            temp_path = None
            
            # Detect language
            logger.info("Detecting language...")
            audio_array = whisper.pad_or_trim(audio_array)
            mel = whisper.log_mel_spectrogram(audio_array).to(whisper_model.device)
            _, probs = whisper_model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            confidence = probs[detected_language]
            
            logger.info(f"Detected language: {detected_language} (confidence: {confidence:.2f})")
            
            # Transcribe with detected language
            result = whisper_model.transcribe(
                audio_array,
                language=detected_language,
                fp16=False,
                verbose=False
            )
            
            return {
                "text": result["text"].strip(),
                "language": detected_language,
                "confidence": confidence
            }
            
        finally:
            if temp_path and os.path.exists(temp_path):
                safe_remove_file(temp_path)
            
    except Exception as e:
        logger.error(f"Language detection transcription error: {e}")
        raise

# Utility functions remain the same
def get_temp_dir_info() -> dict:
    """Get information about the temp directory"""
    try:
        if not TEMP_DIR.exists():
            return {"exists": False, "file_count": 0, "total_size": 0}
        
        files = list(TEMP_DIR.glob("audio_*.wav"))
        total_size = sum(f.stat().st_size for f in files if f.exists())
        
        return {
            "exists": True,
            "path": str(TEMP_DIR),
            "file_count": len(files),
            "total_size": total_size,
            "files": [f.name for f in files]
        }
    except Exception as e:
        logger.error(f"Error getting temp dir info: {e}")
        return {"error": str(e)}

def clear_temp_directory():
    """Clear all files in temp directory"""
    try:
        if not TEMP_DIR.exists():
            return {"success": True, "message": "Temp directory doesn't exist"}
        
        files_removed = 0
        for file_path in TEMP_DIR.glob("audio_*.wav"):
            if safe_remove_file(str(file_path)):
                files_removed += 1
        
        return {
            "success": True,
            "files_removed": files_removed,
            "message": f"Removed {files_removed} files from temp directory"
        }
    except Exception as e:
        logger.error(f"Error clearing temp directory: {e}")
        return {"success": False, "error": str(e)}

# Initialize model when module is imported (optional)
try:
    initialize_whisper_model()
except Exception as e:
    logger.warning(f"Could not initialize Whisper model on import: {e}")
    logger.info("Model will be initialized on first use")