# Simple Vector DB System

This is the simplest implementation of a vector database system that supports:
- Dense embeddings for English using `all-MiniLM-L6-v2`.
- Hybrid target search (Dense + Sparse) for Vietnamese using `bge-m3`.
- Local embedded Qdrant database.
- FastAPI endpoints for immediate indexing and querying.

## Setup
Create a virtual environment and install the requirements:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Testing Components
We have included dedicated scripts to test embedding and search quality endpoints:

1. **Test Embedding Quality**:
```bash
python test_embeddings.py
```
This script evaluates the model loading mechanism, outputs vector dimensionality, and ensures both dense and sparse encodings work properly.


2. **Test Search Pipeline**:
```bash
python test_search.py
```
This tests Qdrant locally using `./qdrant_data`. It indexes sample sentences and queries them using native Hybrid RRF (Reciprocal Rank Fusion) search for Vietnamese, and standard Cosine for English.


## Running the API
The endpoints are built with FastAPI. Start the API locally:
```bash
python api.py
# or using uvicorn:
# uvicorn api:app --reload
```

## Make API Requests

### 1. Indexing Document (Vietnamese)
```bash
curl -X POST "http://localhost:8000/index" \
     -H "Content-Type: application/json" \
     -d '{"text": "Trí tuệ nhân tạo sẽ thay đổi thế giới máy tính.", "lang": "vi"}'
```

### 2. Search Query
```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "trí tuệ và máy tính", "lang": "vi", "top_k": 3}'
```