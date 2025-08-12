from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
import uuid
import os
import asyncio
from datetime import datetime
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
initialization_error = None

async def initialize_models_async():
    """Initialize models asynchronously after server starts"""
    global models_initialized, initialization_error
    try:
        logger.info("🤖 Initializing models in background...")
        
        # Check if running on Render (or other cloud platform)
        if not os.path.exists("models/vosk-model-small-en-us-0.15"):
            logger.info("📥 Vosk model not found. Downloading...")
            # Download and extract model
            import urllib.request
            import zipfile
            
            os.makedirs("models", exist_ok=True)
            model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, urllib.request.urlretrieve, model_url, "models/model.zip")
            
            with zipfile.ZipFile("models/model.zip", 'r') as zip_ref:
                zip_ref.extractall("models/")
            
            os.remove("models/model.zip")
            logger.info("✅ Vosk model downloaded and extracted")
        
        # Initialize models in executor to avoid blocking
        logger.info("🔄 Calling initialize_models()...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, initialize_models)
        models_initialized = True
        initialization_error = None
        logger.info("✅ All models initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        models_initialized = False
        initialization_error = str(e)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    logger.info("🔄 Application starting up...")
    logger.info(f"📍 Port: {PORT}")
    logger.info(f"🌍 Environment: {os.environ.get('RENDER', 'local')}")
    
    # Create background task for model initialization
    # Don't await it - let it run in background
    task = asyncio.create_task(initialize_models_async())
    
    yield  # Server starts here
    
    # Cleanup on shutdown
    logger.info("🔄 Application shutting down...")
    if not task.done():
        task.cancel()

# Create FastAPI app
app = FastAPI(
    title="AR Assistant API", 
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
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

# ROOT ENDPOINT - Support both GET and HEAD for Render
@app.get("/")
@app.head("/")
def read_root():
    """Root endpoint - must respond quickly for Render to detect the port"""
    return {
        "status": "online",
        "message": "AR Assistant API is running",
        "port": PORT,
        "models_ready": models_initialized,
        "initialization_error": initialization_error,
        "timestamp": datetime.now().isoformat()
    }

# HEALTH CHECK ENDPOINTS - Support both GET and HEAD
@app.get("/health")
@app.head("/health")
def health_check():
    """Primary health check endpoint"""
    return {
        "status": "healthy",
        "service": "ar-assistant",
        "port": PORT,
        "models_ready": models_initialized,
        "initialization_error": initialization_error,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
@app.head("/api/health")
def api_health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "message": "AR Assistant API is running",
        "models_ready": models_initialized,
        "initialization_error": initialization_error,
        "port": PORT,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/ping")
@app.head("/ping")
def ping():
    """Simple ping endpoint"""
    return {"status": "pong", "port": PORT}

@app.get("/api/system-check")
def system_check():
    """System readiness check"""
    recommendations = []
    system_ready = True
    
    # Check if models are initialized
    if not models_initialized:
        system_ready = False
        if initialization_error:
            recommendations.append(f"Model initialization failed: {initialization_error}")
        else:
            recommendations.append("Models still initializing - please wait")
    
    # Check if required directories exist
    if not os.path.exists("uploads"):
        system_ready = False
        recommendations.append("Create uploads directory")
        
    if not os.path.exists("responses"):
        system_ready = False
        recommendations.append("Create responses directory")
    
    return {
        "system_ready": system_ready,
        "models_initialized": models_initialized,
        "initialization_error": initialization_error,
        "recommendations": recommendations,
        "status": "ready" if system_ready else "initializing",
        "port": PORT,
        "timestamp": datetime.now().isoformat()
    }

# Make all endpoints handle model loading gracefully
@app.post("/api/speech/transcribe")
async def transcribe_speech(audio: UploadFile = File(...)):
    """STT endpoint - transcribe audio to text"""
    try:
        if not models_initialized:
            error_msg = f"Models still initializing"
            if initialization_error:
                error_msg += f": {initialization_error}"
            else:
                error_msg += ", please try again in a few moments"
                
            return JSONResponse(
                status_code=503,
                content={
                    "success": False, 
                    "error": error_msg,
                    "models_ready": False,
                    "initialization_error": initialization_error
                }
            )
            
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
        logger.error(f"Transcription error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/qa/answer")
async def get_answer(question: dict):
    """RAG endpoint - get answer for text question"""
    try:
        if not models_initialized:
            error_msg = f"Models still initializing"
            if initialization_error:
                error_msg += f": {initialization_error}"
            else:
                error_msg += ", please try again in a few moments"
                
            return JSONResponse(
                status_code=503,
                content={
                    "success": False, 
                    "error": error_msg,
                    "initialization_error": initialization_error
                }
            )
            
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
        logger.error(f"RAG error: {e}")
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
        logger.error(f"TTS error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/rag/pipeline")
async def rag_pipeline_endpoint(audio: UploadFile = File(...)):
    """RAG pipeline endpoint - STT + RAG combined"""
    try:
        if not models_initialized:
            error_msg = f"Models still initializing"
            if initialization_error:
                error_msg += f": {initialization_error}"
            else:
                error_msg += ", please try again in a few moments"
                
            return JSONResponse(
                status_code=503,
                content={
                    "success": False, 
                    "error": error_msg,
                    "initialization_error": initialization_error
                }
            )
            
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
        logger.error(f"RAG Pipeline error: {e}")
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
            error_msg = f"Models still initializing"
            if initialization_error:
                error_msg += f": {initialization_error}"
            else:
                error_msg += ", please try again in a few moments"
                
            return JSONResponse(
                status_code=503,
                content={
                    "success": False, 
                    "error": error_msg,
                    "initialization_error": initialization_error
                }
            )
            
        # Validate audio file
        if not audio.content_type or not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        
        # Read audio data
        audio_bytes = await audio.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Step 1: Speech to Text
        logger.info("Step 1: Converting speech to text...")
        stt_start = datetime.now()
        transcription = transcribe(audio_bytes)
        stt_time = (datetime.now() - stt_start).total_seconds() * 1000
        
        if not transcription or transcription.strip() == "":
            raise HTTPException(status_code=400, detail="No speech detected in audio")
        
        logger.info(f"Transcription: {transcription}")
        
        # Step 2: RAG (Get AI response)
        logger.info("Step 2: Getting AI response...")
        rag_start = datetime.now()
        ai_response = get_rag_response(transcription)
        rag_time = (datetime.now() - rag_start).total_seconds() * 1000
        
        logger.info(f"AI Response: {ai_response}")
        
        # Step 3: Text to Speech
        logger.info("Step 3: Converting text to speech...")
        tts_start = datetime.now()
        audio_filename = f"response_{uuid.uuid4()}.wav"
        audio_path = f"responses/{audio_filename}"
        
        # Use async TTS generation
        await async_tts_generate(ai_response, audio_path)
        tts_time = (datetime.now() - tts_start).total_seconds() * 1000
        
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Pipeline completed in {total_time:.0f}ms")
        
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
        logger.error(f"Pipeline error: {e}")
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
                    logger.error(f"Failed to delete {file}: {e}")
        
        # Clear upload files
        if upload_dir.exists():
            for file in upload_dir.glob("*"):
                try:
                    if file.is_file():
                        file.unlink()
                        cleaned_files += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file}: {e}")
        
        return {
            "success": True, 
            "message": f"Cache cleared - {cleaned_files} files deleted",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
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
                            logger.error(f"Failed to process {file}: {e}")
        
        return {
            "success": True,
            "message": f"Cleaned {cleaned_files} old files",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
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
            "initialization_error": initialization_error,
            "response_files": response_files,
            "upload_files": upload_files,
            "port": PORT,
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
            error_msg = f"Models still initializing"
            if initialization_error:
                error_msg += f": {initialization_error}"
            else:
                error_msg += ", please try again in a few moments"
                
            return JSONResponse(
                status_code=503,
                content={
                    "success": False, 
                    "error": error_msg,
                    "initialization_error": initialization_error
                }
            )
            
        audio_bytes = await audio.read()
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
            error_msg = f"Models still initializing"
            if initialization_error:
                error_msg += f": {initialization_error}"
            else:
                error_msg += ", please try again in a few moments"
                
            return JSONResponse(
                status_code=503,
                content={
                    "success": False, 
                    "error": error_msg,
                    "initialization_error": initialization_error
                }
            )
            
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

# For production deployment (Render will use this automatically)
if __name__ == "__main__":
    import uvicorn
    logger.info(f"🚀 Starting AR Assistant API on port {PORT}")
    logger.info("📝 Make sure your rag_engine.py, stt.py, and tts.py modules are available")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        access_log=True,
        log_level="info"
    )
