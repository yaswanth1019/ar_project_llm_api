from vosk import Model, KaldiRecognizer
import io
import wave
import json

# Initialize model once
model = Model("models/vosk-model-small-en-us-0.15")

def transcribe(audio_bytes: bytes) -> str:
    """Transcribe audio bytes to text using Vosk"""
    try:
        # Create BytesIO from audio bytes
        audio_io = io.BytesIO(audio_bytes)
        
        # Open as wave file
        with wave.open(audio_io, "rb") as wf:
            rec = KaldiRecognizer(model, wf.getframerate())
            
            result = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    result += res.get("text", "") + " "
            
            # Get final result
            final_res = json.loads(rec.FinalResult())
            result += final_res.get("text", "")
            
            return result.strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        raise e