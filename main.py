"""
AI Service for Kosera SPK
Provides text embedding generation using Sentence Transformers.

Endpoints:
- GET /health - Health check
- POST /vectorize - Generate embedding for single text
- POST /vectorize/batch - Generate embeddings for multiple texts

Model: paraphrase-multilingual-MiniLM-L12-v2
- Supports 50+ languages including Indonesian
- Output dimension: 384
"""

import os
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Configuration
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", "/app/models")
EMBEDDING_DIMENSION = 384  # Output dimension for multilingual-MiniLM-L12-v2

# Global model storage
ml_models: dict = {}


# ============================================================
# Lifespan Context Manager (Modern Startup/Shutdown)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern lifespan handler for FastAPI.
    Replaces deprecated @app.on_event("startup") and @app.on_event("shutdown")
    """
    # Startup: Load model
    print("=" * 50)
    print("Kosera AI Service Starting...")
    print(f"Loading model: {MODEL_NAME}")
    print(f"Cache directory: {CACHE_DIR}")
    
    try:
        ml_models["encoder"] = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)
        print(f"Model loaded successfully!")
        print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
        print("=" * 50)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model - {e}")
        # Let container fail so orchestrator can restart it
        raise
    
    yield  # Application runs here
    
    # Shutdown: Cleanup
    print("Kosera AI Service Shutting Down...")
    ml_models.clear()
    print("Model unloaded.")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Kosera AI Service",
    description="Text embedding service for semantic search in Kosera SPK",
    version="1.1.0",
    lifespan=lifespan
)

# CORS middleware (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request/Response Models
# ============================================================

class TextRequest(BaseModel):
    """Request model for single text vectorization."""
    text: str = Field(..., min_length=1, description="Text to vectorize")


class BatchTextRequest(BaseModel):
    """Request model for batch text vectorization."""
    texts: List[str] = Field(
        ..., 
        min_length=1, 
        max_length=100, 
        description="List of texts to vectorize (max 100)"
    )


class VectorResponse(BaseModel):
    """Response model for single vector."""
    status: str = "success"
    vector: List[float]
    dimension: int = EMBEDDING_DIMENSION


class BatchVectorResponse(BaseModel):
    """Response model for batch vectors."""
    status: str = "success"
    vectors: List[List[float]]
    count: int
    dimension: int = EMBEDDING_DIMENSION


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model: str
    dimension: int
    ready: bool


# ============================================================
# Endpoints
# ============================================================

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint to verify service is running and model is loaded."""
    model_ready = "encoder" in ml_models
    
    if not model_ready:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Service is starting up."
        )
    
    return HealthResponse(
        status="AI Service is running",
        model=MODEL_NAME,
        dimension=EMBEDDING_DIMENSION,
        ready=model_ready
    )


@app.post("/vectorize", response_model=VectorResponse)
def generate_vector(payload: TextRequest):
    """
    Generate embedding vector for a single text.
    
    The model (paraphrase-multilingual-MiniLM-L12-v2) outputs a 384-dimensional 
    vector that captures the semantic meaning of the input text.
    Supports Indonesian and 50+ other languages.
    """
    if "encoder" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        text = payload.text.strip()
        if not text:
            raise HTTPException(
                status_code=400, 
                detail="Text cannot be empty or whitespace only"
            )
        
        # Generate embedding (CPU-based, fast for short texts)
        vector_embedding = ml_models["encoder"].encode(text).tolist()
        
        return VectorResponse(
            status="success",
            vector=vector_embedding,
            dimension=len(vector_embedding)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Embedding generation failed: {str(e)}"
        )


@app.post("/vectorize/batch", response_model=BatchVectorResponse)
def generate_vectors_batch(payload: BatchTextRequest):
    """
    Generate embedding vectors for multiple texts in a single request.
    
    IMPORTANT: All texts must be non-empty to maintain index alignment.
    If any text is empty, the request will be rejected with 400 error.
    This ensures the returned vectors array matches the input texts array 1:1.
    
    Maximum 100 texts per request.
    """
    if "encoder" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # CRITICAL: Validate all texts to maintain index alignment
        # Do NOT filter out empty texts as it would misalign the response
        for index, text in enumerate(payload.texts):
            if not text.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"Item at index {index} is empty. All batch items must be valid non-empty text."
                )
        
        # Generate embeddings in batch (much faster than individual calls)
        # Index alignment is preserved: vectors[i] corresponds to texts[i]
        vectors = ml_models["encoder"].encode(payload.texts).tolist()
        
        return BatchVectorResponse(
            status="success",
            vectors=vectors,
            count=len(vectors),
            dimension=EMBEDDING_DIMENSION
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Batch embedding generation failed: {str(e)}"
        )


# ============================================================
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
# ============================================================
