import os
import requests
import concurrent.futures
import pandas as pd
import numpy as np
import faiss
import shap
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from snorkel.labeling import labeling_function, PandasLFApplier, LabelModel
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ===========================
# CONFIG
# ===========================
WIKI_URL = "https://en.wikipedia.org/w/api.php"
GPT_MODEL_NAME = "google/mt5-small"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LABEL_CARDINALITY = 3

# ===========================
# FUNCTIONS
# ===========================
def fetch_pages(titles):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
        "titles": "|".join(titles),
    }
    r = requests.get(WIKI_URL, params=params)
    r.raise_for_status()
    return {page['title']: page.get('extract', "") for page in r.json()['query']['pages'].values()}

def batch_fetch(titles, batch_size=5):
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as ex:
        futures = []
        for i in range(0, len(titles), batch_size):
            futures.append(ex.submit(fetch_pages, titles[i:i+batch_size]))
        for future in concurrent.futures.as_completed(futures):
            results.update(future.result())
    return results

def generate_synthetic(data):
    pipe = pipeline("text2text-generation", model=GPT_MODEL_NAME)
    synth = []
    for context in data:
        outputs = pipe(f"summarize: {context}", max_length=100, num_return_sequences=1)
        synth.append(outputs[0]['generated_text'])
    return synth

def build_faiss_index(texts):
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, texts, embedder

@labeling_function()
def lf_positive(x): return 1 if "good" in x.lower() else 0
@labeling_function()
def lf_negative(x): return 2 if "bad" in x.lower() else 0

def weak_supervision(df):
    applier = PandasLFApplier(lfs=[lf_positive, lf_negative])
    L = applier.apply(df)
    label_model = LabelModel(cardinality=LABEL_CARDINALITY, verbose=False)
    label_model.fit(L_train=L, n_epochs=100)
    preds = label_model.predict(L=L)
    df['weak_label'] = preds
    return df

def explain(model_name, texts):
    model = pipeline("sentiment-analysis", model=model_name)
    explainer = shap.Explainer(model)
    shap_values = explainer(texts)
    return shap_values.values.tolist()

# ===========================
# PIPELINE
# ===========================
if __name__ == "__main__":
    # Fetch wiki
    titles = ["Artificial intelligence", "Deep learning"]
    wiki_data = batch_fetch(titles)

    # Synthetic generation
    contexts = list(wiki_data.values())
    synthetic_texts = generate_synthetic(contexts)

    # RAG setup
    index, docs, embedder = build_faiss_index(contexts)

    # Weak supervision
    df = pd.DataFrame({"text": synthetic_texts})
    df = weak_supervision(df)
    print(df)

    # Explain model predictions
    shap_vals = explain("nlptown/bert-base-multilingual-uncased-sentiment", synthetic_texts[:2])
    print("SHAP Values:", shap_vals)

# ===========================
# FASTAPI APP
# ===========================
app = FastAPI()

class InputText(BaseModel):
    input_text: str

gen_pipe = pipeline("text2text-generation", model=GPT_MODEL_NAME)

@app.post("/generate/")
def generate_text(data: InputText):
    output = gen_pipe(data.input_text, max_length=100)
    return {"result": output[0]['generated_text']}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
