from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import torch

# 1. โหลด datasets งานต่าง ๆ มา
dataset_qa = load_dataset("squad", split="train[:10%]")  # ตัวอย่างน้อย ๆ
dataset_summ = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")
dataset_cls = load_dataset("glue", "sst2", split="train[:10%]")

# 2. เตรียมข้อมูลแบบ unified format: 'input', 'target', 'task'
def preprocess_qa(example):
    return {
        "input": "question: " + example["question"] + " context: " + example["context"],
        "target": example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else "",
        "task": "qa"
    }
def preprocess_summ(example):
    return {
        "input": "summarize: " + example["article"],
        "target": example["highlights"],
        "task": "summarization"
    }
def preprocess_cls(example):
    label_map = {0: "negative", 1: "positive"}
    return {
        "input": "sentiment: " + example["sentence"],
        "target": label_map[example["label"]],
        "task": "classification"
    }

dataset_qa = dataset_qa.map(preprocess_qa)
dataset_summ = dataset_summ.map(preprocess_summ)
dataset_cls = dataset_cls.map(preprocess_cls)

# 3. รวม datasets
dataset = concatenate_datasets([dataset_qa, dataset_summ, dataset_cls])

# 4. โหลด tokenizer และ model (ใช้ T5-base เป็นตัวอย่าง)
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 5. Tokenize input และ target
def tokenize_fn(examples):
    model_inputs = tokenizer(examples["input"], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["target"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    # ตัด token pad ออกสำหรับ labels ให้ Trainer คำนวณ loss ถูกต้อง
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
    ]
    return model_inputs

dataset = dataset.map(tokenize_fn, batched=True)

# 6. ตั้ง TrainingArguments และ Trainer
training_args = TrainingArguments(
    output_dir="./mtl_t5",
    evaluation_strategy="steps",
    eval_steps=200,
    logging_steps=100,
    save_steps=500,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=3e-4,
    num_train_epochs=3,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset.select(range(100)),  # eval เล็ก ๆ
)

# 7. Train
trainer.train()

# Multi-task Learning คืออะไร?
# ฝึกโมเดลให้ทำงานหลาย ๆ งาน (tasks) พร้อมกัน

# ทำให้โมเดลเรียนรู้ representation ที่ general ขึ้น

# ช่วยเพิ่มประสิทธิภาพ และลดเวลาการเทรนหลายโมเดล