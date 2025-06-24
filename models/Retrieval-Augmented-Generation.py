!pip install langchain chromadb sentence-transformers transformers

from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import pipeline

# 1. โหลดเอกสาร
loader = DirectoryLoader('./my_docs')
docs = loader.load()

# 2. แปลงเป็นเวกเตอร์
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = Chroma.from_documents(docs, embedding)

# 3. สร้าง retrieval-augmented QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=pipeline(
        "text-generation",
        model="meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
        max_length=256
    ),
    retriever=db.as_retriever(search_kwargs={'k': 3})
)

# 4. ถามคำถาม
answer = qa_chain.run("อธิบายเนื้อหาในเอกสารเกี่ยวกับวิธีการฝึกโมเดลแบบ LoRA")
print(answer)
