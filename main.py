from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import ollama
import os
import asyncio
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Template directory
templates = Jinja2Templates(directory="templates")

# Configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
IMAGE_PROCESSING_TIMEOUT = 60  # seconds
MAX_IMAGE_SIZE_MB = 10  # 10MB maximum file size
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def run_ollama_chat(image_path: str) -> dict:
    """Run Ollama chat in a thread pool executor"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(
            pool,
            lambda: ollama.chat(
                model="llava:13b",
                messages=[
                    {
                        "role": "user",
                        "content": "Give me a simple description of the image in just one sentence.",
                        "images": [image_path]
                    }
                ]
            )
        )

async def process_image_with_timeout(file: UploadFile) -> str:
    """Process image with timeout handling"""
    try:
        # Verify file type
        if file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(ALLOWED_MIME_TYPES)}"
            )

        # Verify file size
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size is {MAX_IMAGE_SIZE_MB}MB"
            )
        
        # Save the uploaded file
        image_path = str(UPLOAD_DIR / file.filename)
        logger.info(f"Saving uploaded file to: {image_path}")
        
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())

        # Verify the image was saved correctly
        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=500,
                detail="Failed to save uploaded file"
            )

        # Get the description from Ollama with timeout
        logger.info("Sending image to Ollama for processing...")
        
        try:
            res = await asyncio.wait_for(
                run_ollama_chat(image_path),
                timeout=IMAGE_PROCESSING_TIMEOUT
            )
            description = res['message']['content']
            logger.info("Successfully received description from Ollama")
            
        except ollama.ResponseError as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise HTTPException(
                status_code=502,
                detail=f"Ollama API error: {str(e)}"
            )
        except asyncio.TimeoutError:
            logger.warning("Image processing timed out")
            raise HTTPException(
                status_code=504,
                detail="Image processing timed out"
            )
        except Exception as e:
            logger.error(f"Unexpected Ollama error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process image: {str(e)}"
            )
        
        # Clean up - remove the uploaded file
        try:
            os.remove(image_path)
            logger.info(f"Removed temporary file: {image_path}")
        except OSError as e:
            logger.warning(f"Failed to remove temporary file: {e}")

        return description
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_image_with_timeout: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during image processing"
        )

@app.post("/describe-image")
async def describe_image(file: UploadFile = File(...)):
    """Endpoint to describe an uploaded image"""
    try:
        logger.info(f"Received image upload request for file: {file.filename}")
        description = await process_image_with_timeout(file)
        return JSONResponse(content={"description": description})
        
    except HTTPException as he:
        logger.error(f"HTTP error in describe-image: {he.detail}")
        return JSONResponse(
            status_code=he.status_code,
            content={"error": he.detail}
        )
    except Exception as e:
        logger.error(f"Unexpected error in describe-image: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "An unexpected error occurred"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=60,
        log_level="info"
    )