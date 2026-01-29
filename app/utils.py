import nltk
from nltk.tokenize import sent_tokenize

#download once
nltk.download("punkt")

def chunk_text(
        text: str,
        max_chars: int  = 800,
        overlap: int = 100
):
    sentences = sent_tokenize(text)
    chunks = []
    current = ""

    for sentence in sentences:
        if (len(current) + len(sentences) <= max_chars):
            current += " " + sentence
        else:
            chunks.append(current.strip())
            current = sentence
        
    if (current):
        chunks.append(current)
        
    #simple overlap
    if (overlap > 0):
        overlapped = []

        for i,chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                prev = chunk
                overlapped.append(prev[-overlap:] + " " + chunk)
        return overlapped
    
    return chunks


