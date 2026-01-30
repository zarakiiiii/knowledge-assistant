import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-large"

class LocalGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        ).to(self.device)

    def generate(self, question: str, contexts: list[str]) -> str:
        "keep prompt tight to avoid OOM"
        sources = "\n\n".join(
        [f"Source {i+1}: {ctx[:800]}" for i, ctx in enumerate(contexts)]
        )

        prompt = (
            "Answer the question ONLY using the sources below. "
            "If the answer is not present, say 'I don't know'.\n\n"
            f"Sources:\n{sources}\n\nQuestion:\n{question}\n\nAnswer:"
        )
    
        inputs = self.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,  #greedy decoding
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

#outputs actually returns in batch for and we want the first generated sequence also we have one prompt so there'll only be one generated seq.


#singleton (load once)
generator = LocalGenerator()

def generate_answer(question: str, contexts: list[str]) -> str:
    return generator.generate(question, contexts)