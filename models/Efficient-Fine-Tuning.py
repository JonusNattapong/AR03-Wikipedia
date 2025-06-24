!pip install transformers peft bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, load_in_4bit=True, device_map="auto"
)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","v_proj"]
)
model = get_peft_model(base_model, lora_config)

# (LoRA, QLoRA) จากนั้น train กับ Trainer ตามปกติ
