from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer, pipeline
import pandas as pd

model_name = "google/tapas-large-finetuned-wtq"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTableQuestionAnswering.from_pretrained(model_name)

table = pd.DataFrame(
    {
        "Country": ["France", "Italy"],
        "Capital": ["Paris", "Rome"]
    }
)

qa = pipeline(
    "table-question-answering",
    model=model,
    tokenizer=tokenizer
)

print(
    qa(
        table=table,
        query="What is the capital of France?"
    )
)
