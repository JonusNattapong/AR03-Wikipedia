# 🤖 NLP End-to-End Production-Ready Pipeline

🧠 NLP Production Pipeline
โครงการนี้เป็น NLP Pipeline แบบ End-to-End ที่ครอบคลุมการดึงข้อมูลจาก Wikipedia API, สร้างชุดข้อมูล Synthetic, จัดการ Weak Supervision, สร้าง RAG (Retrieval-Augmented Generation), และอธิบายผลโมเดล (Explainability) เพื่อทำงาน NLP ในการใช้งานจริง (production-ready)

This project is a **fully automated NLP pipeline** that:
- Fetches data from Wikipedia
- Generates synthetic QA/summarization data
- Builds a Retrieval-Augmented Generation (RAG) index
- Applies weak supervision with Snorkel
- Runs explainable AI (SHAP/LIME) for bias and error analysis
- Serves inference via FastAPI
- Orchestrates workflows using Airflow
- Deploys as containers using Docker Compose

## 🎯 Features
✅ Parallel data fetching from Wikipedia  
✅ Synthetic data augmentation  
✅ Weak supervision for labeling  
✅ RAG-based retrieval (FAISS)  
✅ Explainability with SHAP/LIME  
✅ FastAPI for real-time inference  
✅ Airflow DAGs for scheduling & automation  
✅ Scalable and cloud-ready with Docker

---

## 🗂️ Project Structure

```

project/
├── app/
│   ├── main.py            # FastAPI service & pipeline
│   ├── requirements.txt   # Application dependencies
│   ├── config.py           # Application config
│   ├── utils.py            # Shared utility functions
│   ├── rag.py              # RAG setup
│   ├── supervision.py      # Weak supervision logic
│   ├── synthetic.py        # Synthetic data generation
│   ├── explain.py          # SHAP explainability
├── airflow/
│   ├── dags/
│   │   └── nlp\_pipeline.py # Airflow DAGs for automation
│   ├── requirements.txt    # Airflow-specific dependencies
│   ├── Dockerfile
├── docker/
│   ├── Dockerfile          # Application image
│   ├── docker-compose.yml  # Compose setup for app + Airflow
├── data/                   # Input & output data
├── .env                    # Secrets and credentials
├── README.md               # Documentation

````

---

## 🐳 Getting Started

### 1️⃣ Clone the repo
```bash
git clone https://github.com/your_user/your_project.git
cd your_project
````

### 2️⃣ Build & Run with Docker Compose

```bash
cd docker
docker-compose up --build
```

### 3️⃣ Test the API

Once up, go to:

```
http://localhost:8000/docs
```

You can POST a request to `/generate/` to get generated output.

### 4️⃣ Access Airflow UI

```
http://localhost:8080
```

(default username/password: `airflow`/`airflow`)
Trigger the `nlp_pipeline` DAG to orchestrate the pipeline.

---

## 🧠 Usage

**Example API request**:

```bash
curl -X POST http://localhost:8000/generate/ \
  -H "Content-Type: application/json" \
  -d '{"input_text": "Please summarize AI"}'
```

**Example DAG**: `nlp_pipeline` DAG will:

* Fetch Wikipedia articles
* Generate synthetic data
* Weakly label the data
* Save datasets & embeddings
* Run SHAP explainability
* Serve via FastAPI

---

## 🧪 Running Locally (without Docker)

You can also run the components manually:

1. Install dependencies:

   ```bash
   pip install -r app/requirements.txt
   ```
2. Run the API:

   ```bash
   python app/main.py
   ```
3. Run Airflow:

   ```bash
   airflow standalone
   ```

---

## 🔑 Configuration

Sensitive credentials go into `.env`.
Set up your `WIKIPEDIA_API_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, etc.

---

## 🚀 CI/CD & Deployment

* **CI/CD**: Add GitHub Actions (`.github/workflows/ci.yml`)
* **Cloud deployment**: Push built images to container registries (AWS ECR/GCP Artifact Registry), deploy on Kubernetes or ECS
* **Scaling**: Horizontal scaling via container orchestrators
* **Monitoring**: Add Prometheus/Grafana for metrics and alerting

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests for enhancements, bug fixes, or new features.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

💡 **Have questions?**
Reach out via GitHub Issues or Discussions — we’d love your feedback!