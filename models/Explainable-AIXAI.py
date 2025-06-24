import shap
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# เตรียม SHAP explainer
explainer = shap.Explainer(nlp, tokenizer)
text = ["ฉันชอบอาหารจังแฮง"]
shap_values = explainer(text)
shap.plots.text(shap_values[0])

# อธิบายทำไมโมเดลถึงตัดสินใจเช่นนั้น เพื่อความโปร่งใสและตรวจสอบได้