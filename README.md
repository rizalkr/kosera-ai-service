---
title: Kosera AI Service
emoji: 🏠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Kosera AI Service

Backend AI untuk Sistem Pendukung Keputusan (SPK) pencarian kos.

## Features

- **Text Embedding**: Generate 384-dimensional vectors untuk semantic search
- **Multilingual**: Support 50+ bahasa termasuk Bahasa Indonesia
- **Fast**: Optimized dengan FastAPI + Sentence Transformers

## Model

**paraphrase-multilingual-MiniLM-L12-v2**
- Dimensions: 384
- Languages: 50+
- Use case: Semantic similarity, text search

## API Endpoints

### Health Check
```
GET /health
```

### Vectorize Single Text
```
POST /vectorize
Content-Type: application/json

{
  "text": "kos murah dekat kampus dengan wifi"
}
```

### Vectorize Batch
```
POST /vectorize/batch
Content-Type: application/json

{
  "texts": ["text 1", "text 2", "text 3"]
}
```

## Response Format

```json
{
  "vector": [0.123, -0.456, ...],
  "dimension": 384,
  "status": "success"
}
```

## Tech Stack

- Python 3.11
- FastAPI
- Sentence Transformers
- Docker

## Related

- Main App: [Kosera SPK](https://github.com/rizalkr/kosera-spk)