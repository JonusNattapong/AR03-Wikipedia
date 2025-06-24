from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# สร้าง prompt แบบ chain-of-thought
question = "If there are 3 boxes with 4 apples each, and I take 5 apples away, how many apples do I have left? Let's reason step-by-step."
prompt = f"{question}\nLet's think step by step."

# pipeline text2text
gen = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer
)

print(gen(prompt, max_length=128)[0]['generated_text'])
