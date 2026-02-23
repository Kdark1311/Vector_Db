from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from qdrant_pipeline import QdrantPipeline

app = FastAPI(title="Vector DB API")

# Initialize the pipeline (will lock and load models via initialization)
pipeline = None

@app.on_event("startup")
def startup_event():
    global pipeline
    print("Initializing API & Qdrant Pipeline...")
    pipeline = QdrantPipeline(storage_path="./qdrant_data")
    print("Initialization complete. Models are ready!")

class DocumentInput(BaseModel):
    text: str
    lang: str = "en"
    metadata: Optional[dict] = None

class SearchInput(BaseModel):
    query: str
    lang: str = "en"
    top_k: int = 5

@app.post("/index")
def index_document(doc: DocumentInput):
    """API Endpoint to embed and index a document."""
    try:
        point_id = pipeline.index_document(doc.text, doc.lang, doc.metadata)
        return {"status": "success", "point_id": point_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search(query: SearchInput):
    """API Endpoint to execute a standard/hybrid search."""
    try:
        results = pipeline.search(query.query, query.lang, query.top_k)
        
        # Format Qdrant ScoredPoint results
        formatted_results = [
            {
                "id": str(r.id),
                "score": r.score,
                "text": r.payload.get("text", ""),
                "metadata": r.payload
            }
            for r in results
        ]
        return {"status": "success", "results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
