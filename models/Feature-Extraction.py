from transformers import AutoModel, AutoTokenizer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

inputs = tokenizer(
    "This is a test",
    return_tensors="pt"
)
outputs = model(**inputs)
embeddings = outputs.last_hidden_state.mean(dim=1)
print(embeddings.shape)  # torch.Size([1, 768])
