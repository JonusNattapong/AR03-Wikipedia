from openprompt import PromptDataLoader, PromptForClassification
from openprompt.prompts import ManualTemplate
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

template = ManualTemplate(
    text='{"placeholder":"text_a"} ผลลัพธ์คือ {"mask"}.',
    tokenizer=tokenizer
)
prompt_model = PromptForClassification(template=template, plm=model, tokenizer=tokenizer)
dataloader = PromptDataLoader(dataset=my_dataset, template=template, tokenizer=tokenizer)

# train prompt embeddings หรือ search prompt tokens

# ออกแบบ prompt ให้โมเดล LLM ตอบได้ดีขึ้น หรือใช้เทคนิคอัตโนมัติ (AutoPrompt, Prompt Tuning)