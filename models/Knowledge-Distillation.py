import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# โหลด teacher และ student โมเดล
teacher_name = "bert-large-uncased"
student_name = "distilbert-base-uncased"

teacher = AutoModelForSequenceClassification.from_pretrained(teacher_name)
student = AutoModelForSequenceClassification.from_pretrained(student_name)
tokenizer = AutoTokenizer.from_pretrained(student_name)

optimizer = optim.Adam(student.parameters(), lr=5e-5)
criterion = nn.KLDivLoss(reduction="batchmean")

def distillation_loss(y_student, y_teacher, temperature=2.0):
    p_student = nn.functional.log_softmax(y_student/temperature, dim=1)
    p_teacher = nn.functional.softmax(y_teacher/temperature, dim=1)
    return criterion(p_student, p_teacher) * (temperature ** 2)

# ตัวอย่าง training loop (สมมติ input)
texts = ["I love NLP", "This is a test"]
inputs = tokenizer(texts, padding=True, return_tensors="pt")

with torch.no_grad():
    teacher_outputs = teacher(**inputs).logits

student.train()
optimizer.zero_grad()
student_outputs = student(**inputs).logits
loss = distillation_loss(student_outputs, teacher_outputs)
loss.backward()
optimizer.step()
print("Distillation loss:", loss.item())
