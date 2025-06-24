from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2",
    device_map="auto"
)

# สร้างคำถาม
questions = [generator("ถามคำถามเกี่ยวกับการเขียนโปรแกรม Python:", max_length=50, num_return_sequences=1)[0]['generated_text']
             for _ in range(50)]

# สร้างคำตอบ
qa_pairs = []
for q in questions:
    ans = generator(f"{q}\nคำตอบ:", max_length=50, num_return_sequences=1)[0]['generated_text']
    qa_pairs.append({"question": q, "answer": ans})

# บันทึก JSONL
import json
with open('qa_pairs.jsonl', 'w', encoding='utf-8') as f:
    for pair in qa_pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")

print(f"Self-Play dataset created: {len(qa_pairs)} pairs")
