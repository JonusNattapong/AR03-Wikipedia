!pip install trl transformers datasets

from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# โหลด preference data
dataset = load_dataset('your/dataset', split='train')

# กำหนด DPO config
config = DPOConfig(
    model_name=model_name,
    learning_rate=5e-5,
    per_device_train_batch_size=2
)

# สร้าง DPO Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=model,  # model reference เดิม
    args=config,
    train_dataset=dataset
)

trainer.train()
trainer.save_model("./dpo_model")
