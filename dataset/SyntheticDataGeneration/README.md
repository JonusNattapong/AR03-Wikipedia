# 🤖 DeepSeek-Driven NLP Dataset Generator

This module provides a **schema-based, extensible NLP dataset and prompt generation system** using only the DeepSeek API (no Hugging Face or local models). It supports multiple NLP tasks (including Thai), JSONL/CSV export, and both web (Gradio) and REST API interfaces. Wikipedia can be used as a knowledge base for context via DeepSeek function calling.

---

## 🚀 Features

- **DeepSeek API only**: No Hugging Face or local model dependencies
- **Schema-driven**: Easily extensible for new NLP tasks and fields
- **Supports Thai and multilingual tasks**
- **Wikipedia integration**: Use Wikipedia content as context for dataset generation
- **Gradio web UI**: Select task, number of rows, export format, API key, model, temperature, and Wikipedia query
- **REST API**: FastAPI endpoints for programmatic access
- **Robust error handling**: Retries, JSON decode fixes, and user feedback
- **Export**: JSONL (with `ensure_ascii=False` for readable Thai) and CSV
- **Task display**: View all available tasks in the UI
- **No local model code**: All generation is via DeepSeek API

---

## 🗂️ Structure

```
dataset/SyntheticDataGeneration/
├── main.py         # Main logic, DeepSeek API, schema, Gradio, Wikipedia, export
├── config.py       # (Optional) Configurations
├── utils.py        # (Optional) Utilities
├── rag.py          # (Optional) RAG logic
├── supervision.py  # (Optional) Weak supervision
├── synthetic.py    # (Optional) Synthetic data helpers
├── explain.py      # (Optional) Explainability
├── requirements.txt
├── README.md       # (This file)
```

---

## 🧑‍💻 Usage

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Set your DeepSeek API key

Create a `.env` file with:
```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

Or provide the key in the Gradio UI.

### 3️⃣ Run the Gradio web interface

```bash
python main.py
```

- Select your task, number of rows, export format, and Wikipedia query.
- Enter your DeepSeek API key and select model/temperature.
- Click **Generate and Export**.
- Download the generated dataset (JSONL or CSV, with proper Thai encoding).

### 4️⃣ REST API (FastAPI)

You can also run the FastAPI server for programmatic access (see `main.py` for details).

---

## 📝 Example Tasks

- Text Classification (Thai)
- Summarization
- Question Answering
- Translation (Thai dialects, multilingual)
- Token Classification
- Table QA
- Multiple Choice
- Text Generation
- Sentence Similarity
- ...and more (see UI for full list)

---

## 🌏 Wikipedia Integration

- Enter a Wikipedia query in the UI to use Wikipedia content as context for dataset generation.
- The system will fetch and inject relevant Wikipedia text into the DeepSeek prompt.

---

## ⚡ Export

- **JSONL**: Unicode/Thai preserved (`ensure_ascii=False`)
- **CSV**: UTF-8 encoding

---

## 🛡️ Error Handling

- Retries on DeepSeek API errors
- JSON decode error fixes (auto-extracts JSON from model output)
- User feedback in UI for errors and warnings

---

## 🧩 Extending

- Add new tasks or fields by editing the `SCHEMA` in `main.py`
- UI and API will reflect new tasks automatically

---

## 📜 License

MIT License

---

**Questions?** Open an issue or discussion!