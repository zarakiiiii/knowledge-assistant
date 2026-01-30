#convert text -> embeddings, store embeddings in FAISS, search by semantic similarity, get back original text




INDEX_PATH = "data/index/faiss.index"
TEXT_PATH = "data/index/text_chunks.json"

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json

os.makedirs("data/index", exist_ok=True)

class VectorStore:
    def __init__(self, dim: int = 384):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  #sentence embedding model
        self.index = faiss.IndexFlatL2(dim)  #nearest neighbor search
        self.text_chunks = []
        self.index.reset()

    def add_texts(self, texts: list[str]):
        embeddings = self.model.encode(texts, convert_to_numpy=True)  
        self.index.add(embeddings)
        self.text_chunks.extend(texts)
    
    #query --> user question , k-> top k similar chunks
    def search(self, query: str, k: int = 5):
        if len(self.text_chunks) == 0:
            return []
        
        query_emb = self.model.encode([query], convert_to_numpy=True)
        
        #avoiding asking FAISS for more than we have
        k=min(k, len(self.text_chunks))   
        #query wrapped in list because model expects batch i/p
        distances, indices = self.index.search(query_emb, k)
        #indices=posn of closest vectors, distances=how far they are

        results = []
        for dist,idx in zip(distances[0],indices[0]):
            if idx < len(self.text_chunks):
                results.append({
                    "text": self.text_chunks[idx],
                    "distance": float(dist)
                })
        return results
    
    def save(self):
        faiss.write_index(self.index, INDEX_PATH)
        #self.index is faiss vector index
        #we are saving embeddings instead of recomputing everytime the app is restarted
        with open(TEXT_PATH, "w", encoding = "utf-8") as f:
            json.dump(self.text_chunks, f)

    def load(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(TEXT_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            #reads the binary faiss file & recreates the vector index exactly as it was
            with open(TEXT_PATH, "r", encoding="utf-8") as f:
                self.text_chunks = json.load(f)    
     

     #faiss only stores vectors in the order we add them, index.add(embeddings) gives this structure: vecA->chunkA, vecB->chunkB, etc.

     #indices return both distance, index ---> eg: [[2,0,3]] closest at 2, 2nd closest at 0, third closest at 3 and so on
     #indices[0] = [2,0,3]

     #distances = the distances eg: [[0.21,0.47,0.89]]
