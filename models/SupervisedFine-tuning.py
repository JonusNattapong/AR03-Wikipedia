from transformers import AutoModelForQuestionAnswering, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset

model_name = "bert-base-uncased"
dataset = load_dataset("squad")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def preprocess(example):
    return tokenizer(example['question'], example['context'], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="./qa_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation']
)

trainer.train()
