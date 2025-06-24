import pandas as pd
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
import evaluate

# ===========================
# 1) Load Dataset
# ===========================
dataset = load_dataset(
    "json",
    data_files="dataset.jsonl"
)
dataset = dataset['train']

# ===========================
# 2) Encode labels
# ===========================
unique_labels = list(set(dataset['topic']))
unique_labels.sort()
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

def encode_labels(example):
    example['label'] = label2id[example['topic']]
    return example

dataset = dataset.map(encode_labels)

# ===========================
# 3) Train/Test Split
# ===========================
splits = dataset.train_test_split(test_size=0.2, seed=42)
train_set, test_set = splits['train'], splits['test']

# ===========================
# 4) Tokenizer
# ===========================
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(
        example['text'],
        padding=False,
        truncation=True,
        max_length=256
    )

train_set = train_set.map(tokenize_function, batched=True)
test_set = test_set.map(tokenize_function, batched=True)

# ===========================
# 5) Data collator
# ===========================
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

# ===========================
# 6) Load Model
# ===========================
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)

# ===========================
# 7) Metrics
# ===========================
accuracy = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

# ===========================
# 8) Trainer
# ===========================
training_args = TrainingArguments(
    output_dir="./checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=test_set,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ===========================
# 9) Train
# ===========================
trainer.train()

# ===========================
# 10) Save Model
# ===========================
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("✅ Training complete. Model saved to ./fine_tuned_model")
