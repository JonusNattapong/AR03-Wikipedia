from transformers import pipeline

clf = pipeline(
    "zero-shot-classification",
    model="xlm-roberta-large",
    tokenizer="xlm-roberta-large"
)
text = "เฮ็ดจังได๋ให้น้ำซุปแซ่บหลาย"
labels = ["cooking", "health", "travel"]
result = clf(text, candidate_labels=labels)
print(result)

# เทคนิคภาษาถิ่น: รวมภาษาถิ่นเข้าเป็น “pseudo-token” เช่น <isan> 
# แล้ว fine-tune mT5 ให้ generate คำแปลภาษากลาง ↔ ภาษาถิ่น