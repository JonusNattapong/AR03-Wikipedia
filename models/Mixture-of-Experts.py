from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# pipeline chat
chat = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=200
)

# ลองถามโมเดล
question = "Explain the concept of mixture-of-experts architecture."
response = chat(
    f"<s>[INST] {question} [/INST]"
)
print(response[0]['generated_text'])
