from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer
)

print(
    summarizer(
        "The moon is Earth's only natural satellite. It is the fifth-largest satellite in the Solar System and the largest and most massive relative to its parent planet.",
        max_length=50,
        min_length=25
    )
)
