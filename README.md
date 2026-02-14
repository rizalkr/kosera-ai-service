# Kosera AI Service

Python microservice for text embedding generation using Sentence Transformers.

## Overview

This service provides semantic text embedding capabilities for the Kosera SPK (Decision Support System). It uses the `paraphrase-multilingual-MiniLM-L12-v2` model to convert text descriptions into 384-dimensional vectors for similarity search.

## Model Information

- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Size**: ~480MB
- **Dimension**: 384
- **Languages**: 50+ languages including **Indonesian**, English, etc.
- **Speed**: ~10ms per text (CPU)
- **Use Case**: Multilingual semantic similarity, text search

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Health check |
| POST | `/vectorize` | Generate embedding for single text |
| POST | `/vectorize/batch` | Generate embeddings for multiple texts (max 100) |

## API Examples

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "AI Service is running",
  "model": "paraphrase-multilingual-MiniLM-L12-v2",
  "dimension": 384,
  "ready": true
}
```

### Single Text Vectorization

```bash
curl -X POST http://localhost:8000/vectorize \
  -H "Content-Type: application/json" \
  -d '{"text": "Kos nyaman dekat kampus dengan WiFi dan AC"}'
```

Response:
```json
{
  "status": "success",
  "vector": [0.123, -0.456, ...],
  "dimension": 384
}
```

### Batch Vectorization

```bash
curl -X POST http://localhost:8000/vectorize/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Kos murah dekat kampus", "Kos mewah dengan kolam renang"]}'
```

Response:
```json
{
  "status": "success",
  "vectors": [[0.123, ...], [0.456, ...]],
  "count": 2,
  "dimension": 384
}
```

**IMPORTANT**: Batch endpoint maintains index alignment. All texts must be non-empty.
If any text is empty, the request will be rejected with 400 error.

## Local Development

### Without Docker

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download model (first time only, ~480MB)
python download_model.py

# Run the service
uvicorn main:app --reload --port 8000
```

### With Docker

```bash
# Build image (includes model download)
docker build -t kosera-ai-service .

# Run container
docker run -p 8000:8000 kosera-ai-service
```

## Architecture

```
Next.js App (Port 3000)
       |
       | HTTP POST /vectorize
       v
AI Service (Port 8000)
       |
       | Sentence Transformers (multilingual)
       v
384-dim Vector
       |
       | Store in PostgreSQL
       v
pgvector (cosine similarity search)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSFORMERS_CACHE` | `/app/models` | Model cache directory |
| `HF_HOME` | `/app/models` | Hugging Face home directory |

## Why Multilingual Model?

The `paraphrase-multilingual-MiniLM-L12-v2` model is chosen because:

1. **Indonesian Support**: Native support for Bahasa Indonesia
2. **Cross-lingual**: Can match Indonesian query with English description and vice versa
3. **Same Dimension**: Still outputs 384 dimensions (same as MiniLM-L6-v2)
4. **Good Performance**: Slightly larger but excellent quality for multilingual use cases

## Error Handling

| Status Code | Meaning |
|-------------|---------|
| 200 | Success |
| 400 | Bad Request (empty text, invalid input) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (model not loaded) |
