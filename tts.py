from gtts import gTTS
import asyncio
import tempfile
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TTS Configuration
TTS_LANG = 'en'
TTS_SLOW = False

def tts_generate(text: str, output_file: str) -> str:
    """Generate speech from text using Google TTS"""
    try:
        # Validate input
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Limit text length to avoid very long audio files
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        logger.info(f"Generating TTS for text: {text[:50]}...")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create gTTS object with optimized settings
        tts = gTTS(
            text=text, 
            lang=TTS_LANG, 
            slow=TTS_SLOW,
            tld='com'  # Use .com domain for consistency
        )
        
        # Save to file
        tts.save(output_file)
        
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logger.info(f"TTS file created: {output_file}")
            return output_file
        else:
            raise RuntimeError("Audio file not created or is empty")
            
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        raise

async def async_tts_generate(text: str, output_file: str) -> str:
    """Async version of TTS generation"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, tts_generate, text, output_file)

# Legacy function for compatibility
def synthesize(text: str, path: str = "response.wav") -> str:
    """Synchronous wrapper for TTS generation"""
    return tts_generate(text, path)