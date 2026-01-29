from fastapi import FastAPI, UploadFile, File
import os
from app.ingest import extract_text_from_pdf
from app.utils import chunk_text

app = FastAPI(title="Knowledge Assistant")

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

    return {
        "message": "File uploaded succesfully",
        "filename": file.filename,
        "text_preview": text[:300],
        "num_chunks": len(chunks),
        "sample_chunk": chunks[0][:300] if chunks else ""
    }
