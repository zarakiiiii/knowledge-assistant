from fastapi import FastAPI, UploadFile, File
import os
from app.ingest import extract_text_from_pdf
from app.utils import chunk_text
from app.vector_store import VectorStore
from pydantic import BaseModel
from app.generator import generate_answer


app = FastAPI(title="Knowledge Assistant")

vector_store = VectorStore()
vector_store.load()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query_docs(req: QueryRequest):
    results = vector_store.search(req.question, k=5)
    
    #no retrieval
    if not results:
        return {"answer": "I don't know", "sources": []}
    
    #confidence threshold (hallucination control)
    best_distance = results[0]["distance"]
    THRESHOLD = 0.8

    if best_distance > THRESHOLD:
        return {
            "question": req.question,
            "answer": "I don't know",
            "sources": []
        }
    
    #extract only text for generation
    contexts = [item["text"] for item in results]
    
    answer = generate_answer(req.question, contexts)

    return {
        "question": req.question,
        "answer": answer,
        "sources": results
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())  
     #file->UploadFile, file.file -> underlying file-like object, .read() -> reads all bytes into memory 
     
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    vector_store.add_texts(chunks)
    vector_store.save()

    return {
        "message": "File uploaded succesfully",
        "filename": file.filename,
        "text_preview": text[:300],
        "num_chunks": len(chunks),
        "sample_chunk": chunks[0][:300] if chunks else ""
    }
