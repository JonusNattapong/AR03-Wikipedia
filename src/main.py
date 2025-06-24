import json
from transformers import pipeline
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_dataset

# ===========================
# 1) Fetch multiple topics
# ===========================
TOPICS = [
    "Thai culture",
    "Thai cuisine",
    "Thai history",
    "Thai politics"
]

all_docs = []
for topic in TOPICS:
    loader = WikipediaLoader(query=topic, load_max_docs=1, lang="en")
    all_docs.extend(loader.load())

# ===========================
# 2) Split into chunks
# ===========================
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

# ===========================
# 3) Load Pipelines
# ===========================
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# ===========================
# 4) Generate Dataset
# ===========================
candidate_labels = ["history", "culture", "economy", "politics"]
dataset_rows = []

for i, chunk in enumerate(chunks):
    text = chunk.page_content

    summary = summarizer(
        text, max_length=60, min_length=20, do_sample=False
    )[0]['summary_text']

    sentiment_label = sentiment(text)[0]['label']

    topic_label = zero_shot(
        text, candidate_labels=candidate_labels
    )['labels'][0]

    answer = qa(
        question="What is this text about?", context=text
    )['answer']

    named_entities = ner(text)
    fill_mask_pred = fill_mask(
        "This is a [MASK] example."
    )[0]['token_str']

    dataset_rows.append(
        {
            "id": i,
            "text": text,
            "summary": summary,
            "sentiment": sentiment_label,
            "topic": topic_label,
            "qa_answer": answer,
            "entities": named_entities,
            "fill_mask_sample": fill_mask_pred,
        }
    )

# ===========================
# 5) Save to JSONL
# ===========================
output_path = "dataset.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for row in dataset_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(dataset_rows)} examples to {output_path}")

# ===========================
# 6) Load JSONL with Datasets and split
# ===========================
dataset = load_dataset(
    "json",
    data_files=output_path,
    split="train"
)

# train/test split
splits = dataset.train_test_split(test_size=0.2, seed=42)
train_set, test_set = splits["train"], splits["test"]
print(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")

# ===========================
# 7) Ready for fine-tuning
# ===========================
# Example fine-tuning setup:
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
# (Continue with tokenization & Trainer setup...)
