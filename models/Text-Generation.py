from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

inputs = tokenizer(
    "Once upon a time",
    return_tensors="pt"
)
outputs = model.generate(
    **inputs,
    max_length=50,
    num_return_sequences=1
)
print(
    tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
)
