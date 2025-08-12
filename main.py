from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
import uuid
import os
from datetime import datetime
from pathlib import Path

# Get port from environment - this is critical for Render
PORT = int(os.environ.get("PORT", 10000))
print(f"🚀 Starting server on port {PORT}")


# Import your modules
from rag_engine import build_rag_pipeline, get_rag_response, initialize_models
from stt import transcribe
from tts import synthesize, tts_generate, async_tts_generate
from dotenv import load_dotenv

load_dotenv()

# Global flag to track if models are initialized
models_initialized = False

# Update the model initialization to handle cloud deployment
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup
    global models_initialized
    print("Initializing models...")
    try:
        # Check if running on Render (or other cloud platform)
        if not os.path.exists("models/vosk-model-small-en-us-0.15"):
            print("Vosk model not found. Downloading...")
            # Download and extract model
            import urllib.request
            import zipfile
            
            os.makedirs("models", exist_ok=True)
            model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
            urllib.request.urlretrieve(model_url, "models/model.zip")
            
            with zipfile.ZipFile("models/model.zip", 'r') as zip_ref:
                zip_ref.extractall("models/")
            
            os.remove("models/model.zip")
            print("✓ Vosk model downloaded and extracted")
        
        initialize_models()
        models_initialized = True
        print("✓ All models initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize models: {e}")
        models_initialized = False
    
    yield
    
    # Shutdown (cleanup if needed)
    print("Application shutting down...")

app = FastAPI(
    title="AR Assistant API", 
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("responses", exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "AR Assistant API is running", "timestamp": datetime.now().isoformat()}

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "AR Assistant API is running",
        "models_ready": models_initialized,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/system-check")
def system_check():
    """System readiness check"""
    recommendations = []
    system_ready = True
    
    # Check if models are initialized
    if not models_initialized:
        system_ready = False
        recommendations.append("Models not initialized - restart the server")
    
    # Check if required directories exist
    if not os.path.exists("uploads"):
        system_ready = False
        recommendations.append("Create uploads directory")
        
    if not os.path.exists("responses"):
        system_ready = False
        recommendations.append("Create responses directory")
    
    # Check if required modules can be imported
    try:
        from rag_engine import build_rag_pipeline, get_rag_response, initialize_models
        from stt import transcribe
        from tts import synthesize, tts_generate
    except ImportError as e:
        system_ready = False
        recommendations.append(f"Missing required modules: {e}")
    
    return {
        "system_ready": system_ready,
        "models_initialized": models_initialized,
        "recommendations": recommendations,
        "status": "ready" if system_ready else "not_ready",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/speech/transcribe")
async def transcribe_speech(audio: UploadFile = File(...)):
    """STT endpoint - transcribe audio to text"""
    try:
        if not models_initialized:
            raise HTTPException(status_code=503, detail="Models not initialized")
            
        if not audio.content_type or not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        
        audio_bytes = await audio.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        transcription = transcribe(audio_bytes)
        
        return {
            "success": True,
            "transcription": transcription,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Transcription error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/qa/answer")
async def get_answer(question: dict):
    """RAG endpoint - get answer for text question"""
    try:
        if not models_initialized:
            raise HTTPException(status_code=503, detail="Models not initialized")
            
        if "question" not in question:
            raise HTTPException(status_code=400, detail="Question field required")
        
        answer = get_rag_response(question["question"])
        
        return {
            "success": True,
            "question": question["question"],
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"RAG error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/speech/synthesize")
async def synthesize_speech(text_data: dict):
    """TTS endpoint - convert text to speech"""
    try:
        if "text" not in text_data:
            raise HTTPException(status_code=400, detail="Text field required")
        
        text = text_data["text"]
        audio_filename = f"response_{uuid.uuid4()}.wav"
        audio_path = f"responses/{audio_filename}"
        
        # Use async TTS generation
        await async_tts_generate(text, audio_path)
        
        return {
            "success": True,
            "text": text,
            "audioUrl": f"/api/audio/{audio_filename}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"TTS error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/rag/pipeline")
async def rag_pipeline_endpoint(audio: UploadFile = File(...)):
    """RAG pipeline endpoint - STT + RAG combined"""
    try:
        if not models_initialized:
            raise HTTPException(status_code=503, detail="Models not initialized")
            
        if not audio.content_type or not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        
        audio_bytes = await audio.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Use your original pipeline function
        question, answer = build_rag_pipeline(audio_bytes)
        
        return {
            "success": True,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"RAG Pipeline error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/speech/process")
async def process_speech_pipeline(audio: UploadFile = File(...)):
    """Complete speech processing pipeline: STT -> RAG -> TTS"""
    start_time = datetime.now()
    
    try:
        if not models_initialized:
            raise HTTPException(status_code=503, detail="Models not initialized")
            
        # Validate audio file
        if not audio.content_type or not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        
        # Read audio data
        audio_bytes = await audio.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Step 1: Speech to Text
        print("Step 1: Converting speech to text...")
        stt_start = datetime.now()
        transcription = transcribe(audio_bytes)
        stt_time = (datetime.now() - stt_start).total_seconds() * 1000
        
        if not transcription or transcription.strip() == "":
            raise HTTPException(status_code=400, detail="No speech detected in audio")
        
        print(f"Transcription: {transcription}")
        
        # Step 2: RAG (Get AI response)
        print("Step 2: Getting AI response...")
        rag_start = datetime.now()
        ai_response = get_rag_response(transcription)
        rag_time = (datetime.now() - rag_start).total_seconds() * 1000
        
        print(f"AI Response: {ai_response}")
        
        # Step 3: Text to Speech
        print("Step 3: Converting text to speech...")
        tts_start = datetime.now()
        audio_filename = f"response_{uuid.uuid4()}.wav"
        audio_path = f"responses/{audio_filename}"
        
        # Use async TTS generation
        await async_tts_generate(ai_response, audio_path)
        tts_time = (datetime.now() - tts_start).total_seconds() * 1000
        
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        print(f"Pipeline completed in {total_time:.0f}ms")
        
        return {
            "success": True,
            "data": {
                "transcription": transcription,
                "aiResponse": ai_response,
                "audioUrl": f"/api/audio/{audio_filename}"
            },
            "performance": {
                "total_time": int(total_time),
                "stt_time": int(stt_time),
                "rag_time": int(rag_time),
                "tts_time": int(tts_time)
            },
            "metadata": {
                "stt_engine": "vosk",
                "rag_engine": "langchain-groq",
                "tts_engine": "gtts",
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Pipeline error: {e}")
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "processing_time": int(processing_time)
            }
        )

@app.get("/api/audio/{filename}")
def get_audio_file(filename: str):
    """Serve generated audio files"""
    file_path = Path(f"responses/{filename}")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=filename
    )

@app.post("/api/cache/clear")
def clear_cache():
    """Clear response audio files"""
    try:
        response_dir = Path("responses")
        upload_dir = Path("uploads")
        cleaned_files = 0
        
        # Clear response files
        if response_dir.exists():
            for file in response_dir.glob("*.wav"):
                try:
                    file.unlink()
                    cleaned_files += 1
                except Exception as e:
                    print(f"Failed to delete {file}: {e}")
        
        # Clear upload files
        if upload_dir.exists():
            for file in upload_dir.glob("*"):
                try:
                    if file.is_file():
                        file.unlink()
                        cleaned_files += 1
                except Exception as e:
                    print(f"Failed to delete {file}: {e}")
        
        return {
            "success": True, 
            "message": f"Cache cleared - {cleaned_files} files deleted",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Cache clear error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/cleanup")
def cleanup_files():
    """Clean up old files (older than 1 hour)"""
    try:
        response_dir = Path("responses")
        upload_dir = Path("uploads")
        
        current_time = datetime.now().timestamp()
        cleaned_files = 0
        
        for directory in [response_dir, upload_dir]:
            if directory.exists():
                for file in directory.iterdir():
                    if file.is_file():
                        try:
                            file_age = current_time - file.stat().st_mtime
                            if file_age > 3600:  # 1 hour
                                file.unlink()
                                cleaned_files += 1
                        except Exception as e:
                            print(f"Failed to process {file}: {e}")
        
        return {
            "success": True,
            "message": f"Cleaned {cleaned_files} old files",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Cleanup error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# Additional utility endpoints

@app.get("/api/status")
def get_system_status():
    """Get detailed system status"""
    try:
        response_files = len(list(Path("responses").glob("*.wav"))) if Path("responses").exists() else 0
        upload_files = len(list(Path("uploads").glob("*"))) if Path("uploads").exists() else 0
        
        return {
            "system_ready": models_initialized,
            "models_initialized": models_initialized,
            "response_files": response_files,
            "upload_files": upload_files,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/test/stt")
async def test_stt(audio: UploadFile = File(...)):
    """Test STT only"""
    try:
        if not models_initialized:
            raise HTTPException(status_code=503, detail="Models not initialized")
            
        audio_bytes = await audio.read()  # Fixed: Added await
        transcription = transcribe(audio_bytes)
        
        return {
            "success": True,
            "transcription": transcription,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/test/rag")
async def test_rag(question_data: dict):
    """Test RAG only"""
    try:
        if not models_initialized:
            raise HTTPException(status_code=503, detail="Models not initialized")
            
        if "question" not in question_data:
            raise HTTPException(status_code=400, detail="Question field required")
            
        answer = get_rag_response(question_data["question"])
        
        return {
            "success": True,
            "question": question_data["question"],
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/test/tts")
async def test_tts(text_data: dict):
    """Test TTS only"""
    try:
        if "text" not in text_data:
            raise HTTPException(status_code=400, detail="Text field required")
            
        text = text_data["text"]
        audio_filename = f"test_{uuid.uuid4()}.wav"
        audio_path = f"responses/{audio_filename}"
        
        await async_tts_generate(text, audio_path)
        
        return {
            "success": True,
            "text": text,
            "audioUrl": f"/api/audio/{audio_filename}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(    
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# CRITICAL: Make sure the server starts immediately
if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting AR Assistant API on port {PORT}")
    print("📝 Make sure your rag_engine.py, stt.py, and tts.py modules are available")
    
    # Run with explicit configuration
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        access_log=True,
        log_level="info"
    )

# For production deployment (Render will use this)
def create_app():
    """Factory function for production deployment"""
    return app
