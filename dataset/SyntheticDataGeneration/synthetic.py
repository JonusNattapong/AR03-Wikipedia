# app/synthetic.py
from transformers import pipeline
from app.config import LLM_MODEL

def generate_synthetic(texts: list[str], num_return_sequences=1) -> list[str]:
    generator = pipeline("text2text-generation", model=LLM_MODEL)
    results = []
    for text in texts:
        output = generator(
            f"summarize: {text}",
            max_length=100,
            num_return_sequences=num_return_sequences
        )
        results.extend([o['generated_text'] for o in output])
    return results
