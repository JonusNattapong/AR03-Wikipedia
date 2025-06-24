from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)

print(fill_mask("The capital of France is [MASK]."))
