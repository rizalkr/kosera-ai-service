"""
Script to pre-download the Sentence Transformer model during Docker build.
This ensures the model is cached and doesn't need to be downloaded at runtime.
"""

from sentence_transformers import SentenceTransformer
import os

# Using multilingual model for Indonesian language support
# paraphrase-multilingual-MiniLM-L12-v2 supports 50+ languages including Indonesian
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", "/app/models")

def download_model():
    """Download and cache the sentence transformer model."""
    print(f"Downloading model: {MODEL_NAME}")
    print(f"Cache directory: {CACHE_DIR}")
    
    # Download the model (will be cached automatically)
    model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)
    
    # Test the model with sample texts in Indonesian and English
    test_texts = [
        "Test sentence for model verification",
        "Kos nyaman dekat kampus dengan WiFi dan AC",
        "Rumah kost murah di Jakarta Selatan"
    ]
    
    for text in test_texts:
        embedding = model.encode(text)
        print(f"Text: '{text[:40]}...' -> Dimension: {len(embedding)}")
    
    print(f"\nModel downloaded successfully!")
    print(f"Embedding dimension: {len(embedding)}")
    
    return model

if __name__ == "__main__":
    download_model()
