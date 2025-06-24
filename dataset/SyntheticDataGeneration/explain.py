# app/explain.py
import shap
from transformers import pipeline

def explain_predictions(model_name: str, texts: list[str]) -> list:
    clf = pipeline("sentiment-analysis", model=model_name)
    explainer = shap.Explainer(clf)
    shap_values = explainer(texts)
    return shap_values.values.tolist()

# Example usage:
# explain_predictions('nlptown/bert-base-multilingual-uncased-sentiment', ['I love this!', 'I hate this.'])
