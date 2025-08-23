import whisper
import io
import tempfile
import os
import wave
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model once (download happens automatically on first use)
# English-only models (better performance for English):
# tiny.en, base.en, small.en, medium.en
# Multilingual models: tiny, base, small, medium, large, turbo

# For English-only transcription (recommended):
model = None

def initialize_whisper_model():
    """Initialize Whisper model with error handling"""
    global model
    if model is None:
        try:
            logger.info("Loading Whisper model...")
            model = whisper.load_model("base.en")  # Fast + good accuracy for English
            logger.info("✓ Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

def transcribe(audio_bytes: bytes) -> str:
    """Transcribe audio bytes to text using Whisper"""
    try:
        # Initialize model if not already done
        if model is None:
            initialize_whisper_model()
        
        # Validate input
        if not audio_bytes or len(audio_bytes) == 0:
            raise ValueError("Empty audio data")
        
        logger.info(f"Processing audio data of {len(audio_bytes)} bytes")
        
        # Create a temporary file with proper Windows handling
        temp_dir = tempfile.gettempdir()
        logger.info(f"Using temp directory: {temp_dir}")
        
        # Use a more explicit approach for Windows compatibility
        temp_file_path = None
        try:
            # Create temporary file with explicit mode and delete=False
            with tempfile.NamedTemporaryFile(
                mode='wb',
                suffix=".wav", 
                delete=False,
                dir=temp_dir
            ) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
                temp_file.flush()  # Ensure data is written to disk
                os.fsync(temp_file.fileno())  # Force write to disk
            
            logger.info(f"Created temporary file: {temp_file_path}")
            
            # Verify file exists and has content
            if not os.path.exists(temp_file_path):
                raise FileNotFoundError(f"Temporary file not created: {temp_file_path}")
            
            file_size = os.path.getsize(temp_file_path)
            logger.info(f"Temporary file size: {file_size} bytes")
            
            if file_size == 0:
                raise ValueError("Temporary file is empty")
            
            # Try to validate it's a proper audio file (basic check)
            try:
                with wave.open(temp_file_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    logger.info(f"WAV file has {frames} frames")
            except wave.Error:
                # If not a WAV file, that's okay - Whisper can handle various formats
                logger.info("File is not a WAV format, but Whisper should handle it")
            
            # Transcribe the audio file
            logger.info("Starting Whisper transcription...")
            result = model.transcribe(
                temp_file_path,
                fp16=False,  # Disable FP16 to avoid CPU warning
                verbose=False  # Reduce Whisper's verbose output
            )
            
            transcription = result["text"].strip()
            logger.info(f"Transcription completed: '{transcription[:100]}...'")
            
            return transcription
            
        finally:
            # Clean up the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.info("Temporary file cleaned up")
                except OSError as e:
                    logger.warning(f"Could not delete temporary file {temp_file_path}: {e}")
            
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise e

def transcribe_with_fallback(audio_bytes: bytes) -> str:
    """Transcribe with fallback methods for better reliability"""
    try:
        return transcribe(audio_bytes)
    except Exception as first_error:
        logger.warning(f"First transcription attempt failed: {first_error}")
        
        try:
            # Fallback: try with a different temporary directory
            import tempfile
            import shutil
            
            # Try user's temp directory
            user_temp = os.path.expanduser("~/temp")
            if not os.path.exists(user_temp):
                os.makedirs(user_temp, exist_ok=True)
            
            with tempfile.NamedTemporaryFile(
                mode='wb',
                suffix=".wav", 
                delete=False,
                dir=user_temp
            ) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
                temp_file.flush()
                os.fsync(temp_file.fileno())
            
            try:
                if model is None:
                    initialize_whisper_model()
                    
                result = model.transcribe(
                    temp_file_path,
                    fp16=False,
                    verbose=False
                )
                return result["text"].strip()
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as second_error:
            logger.error(f"All transcription attempts failed. First: {first_error}, Second: {second_error}")
            raise Exception(f"Transcription failed: {first_error}")

# Alternative method using BytesIO (sometimes more reliable on Windows)
def transcribe_from_memory_alternative(audio_bytes: bytes) -> str:
    """Alternative transcription method using different approach"""
    try:
        if model is None:
            initialize_whisper_model()
        
        # Save to a specific location instead of temp
        output_dir = os.path.join(os.getcwd(), "temp_audio")
        os.makedirs(output_dir, exist_ok=True)
        
        import uuid
        filename = f"audio_{uuid.uuid4().hex[:8]}.wav"
        file_path = os.path.join(output_dir, filename)
        
        try:
            # Write audio data
            with open(file_path, 'wb') as f:
                f.write(audio_bytes)
            
            # Transcribe
            result = model.transcribe(file_path, fp16=False, verbose=False)
            return result["text"].strip()
            
        finally:
            # Cleanup
            if os.path.exists(file_path):
                os.unlink(file_path)
                
    except Exception as e:
        logger.error(f"Alternative transcription error: {e}")
        raise e

# Example usage with different model sizes and options
def transcribe_with_options(audio_bytes: bytes, model_size="base.en", language="en") -> str:
    """
    Transcribe with custom options
    
    Args:
        audio_bytes: Audio data as bytes
        model_size: Model size (tiny.en, base.en, small.en, medium.en for English only)
        language: Language code (e.g., 'en', 'es', 'fr') or None for auto-detect
    """
    try:
        # Load model if different from global model
        local_model = whisper.load_model(model_size)
        
        with tempfile.NamedTemporaryFile(
            mode='wb',
            suffix=".wav", 
            delete=False
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
            temp_file.flush()
            os.fsync(temp_file.fileno())
        
        try:
            # Transcribe with options
            result = local_model.transcribe(
                temp_file_path,
                language=language,
                task="transcribe",  # or "translate" to translate to English
                fp16=False,
                verbose=False
            )
            return result["text"].strip()
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Custom transcription error: {e}")
        raise e

# Initialize model when module is imported
try:
    initialize_whisper_model()
except Exception as e:
    logger.warning(f"Could not initialize Whisper model on import: {e}")
    logger.info("Model will be initialized on first use")