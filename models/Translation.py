from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

model_name = "Helsinki-NLP/opus-mt-en-th"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer
)

print(translator("Hello, how are you?"))
