from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

qa = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer
)

print(
    qa(
        question="Where does Sarah live?",
        context="My name is Sarah and I live in London."
    )
)
