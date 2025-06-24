from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "mrm8488/t5-small-finetuned-data2text"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

table_data = "name | score\nAlice | 90\nBob | 85"
inputs = tokenizer(
    f"table to text: {table_data}",
    return_tensors="pt"
)

outputs = model.generate(
    **inputs,
    max_length=50
)
print(
    tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
)
