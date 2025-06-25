# AR03-Wikipedia

A production-ready, extensible NLP pipeline for dataset and prompt generation, leveraging the DeepSeek API and Wikipedia as a knowledge base. Supports Thai and multilingual tasks, schema-driven extensibility, and both web (Gradio) and REST API interfaces.

---

## 🚀 Key Features

- **DeepSeek API only**: No Hugging Face or local model dependencies
- **Schema-driven**: Easily add new NLP tasks and fields
- **Wikipedia integration**: Use Wikipedia content as context for dataset generation
- **Supports Thai and multilingual tasks**
- **Gradio web UI**: Interactive dataset generation and export
- **REST API**: FastAPI endpoints for programmatic access
- **Robust error handling**: Retries, JSON decode fixes, and user feedback
- **Export**: JSONL (with readable Thai) and CSV

---

## 🗂️ Project Structure

```
AR03-Wikipedia/
├── app.py
├── README.md                # (This file)
├── requirements.txt
├── dataset/
│   └── SyntheticDataGeneration/
│       ├── main.py          # Main logic, DeepSeek API, schema, Gradio, Wikipedia, export
│       ├── requirements.txt
│       ├── README.md        # Module-specific documentation
│       └── ...
├── models/                  # Model/task scripts (for reference)
├── src/                     # Additional scripts
```

---

## ⚡ Quick Start

1. **Install dependencies**
   ```bash
   pip install -r dataset/SyntheticDataGeneration/requirements.txt
   ```
2. **Set your DeepSeek API key**
   - Create a `.env` file in `dataset/SyntheticDataGeneration/` with:
     ```env
     DEEPSEEK_API_KEY=your_deepseek_api_key
     ```
   - Or enter the key in the Gradio UI.
3. **Run the Gradio web interface**
   ```bash
   python dataset/SyntheticDataGeneration/main.py
   ```
4. **Generate datasets**
   - Select task, number of rows, export format, and Wikipedia query in the UI.
   - Download the generated dataset (JSONL or CSV).

---

## 🌏 Wikipedia Integration

- Enter a Wikipedia query in the UI to use Wikipedia content as context for dataset generation.
- The system fetches and injects relevant Wikipedia text into the DeepSeek prompt.

---

## 🧩 Extending

- Add new tasks or fields by editing the `SCHEMA` in `main.py`.
- UI and API will reflect new tasks automatically.

---

## 📜 License

MIT License

---

**Questions?** Open an issue or discussion!