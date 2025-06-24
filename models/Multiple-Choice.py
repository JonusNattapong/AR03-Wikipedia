from transformers import AutoModelForMultipleChoice, AutoTokenizer
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMultipleChoice.from_pretrained(model_name)

question = "Where is the Eiffel Tower?"
choices = ["London", "Paris", "Rome"]
inputs = tokenizer(
    [question] * len(choices),
    choices,
    padding=True,
    return_tensors="pt"
)
inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}  # batch_size=1

outputs = model(**inputs)
predicted_choice = torch.argmax(outputs.logits, dim=1).item()
print(f"Answer: {choices[predicted_choice]}")
