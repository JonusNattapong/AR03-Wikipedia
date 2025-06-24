from transformers import pipeline

model_name = "facebook/bart-large-mnli"
classifier = pipeline(
    "zero-shot-classification",
    model=model_name
)

print(
    classifier(
        "I am going to a wedding next week.",
        candidate_labels=["sports", "wedding", "politics"]
    )
)
