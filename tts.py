from gtts import gTTS
import os
import asyncio
import io

def synthesize(text: str, path: str = "response.wav") -> str:
    """Synchronous wrapper for TTS generation"""
    return tts_generate(text, path)

def tts_generate(text: str, output_file: str = "response.wav") -> str:
    """Generate speech from text using Google TTS (cloud-friendly)"""
    try:
        # Create gTTS object
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to file
        tts.save(output_file)
        
        if os.path.exists(output_file):
            return output_file
        else:
            raise RuntimeError("Audio file not created.")
    except Exception as e:
        print(f"TTS generation error: {e}")
        raise

async def async_tts_generate(text: str, output_file: str = "response.wav") -> str:
    """Async version of TTS generation"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, tts_generate, text, output_file)