from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
docs = [
    "Bangkok is the capital of Thailand.",
    "Paris is the capital of France.",
    "Tokyo is the capital of Japan."
]

query = "Where is Thailand's capital?"

# สร้าง embeddings
doc_embeds = model.encode(docs, convert_to_tensor=True)
query_embed = model.encode(query, convert_to_tensor=True)

# คำนวณ cosine similarity และดึงอันดับ
scores = util.cos_sim(query_embed, doc_embeds)[0]
ranked_indices = scores.argsort(descending=True).tolist()

for idx in ranked_indices:
    print(f"{docs[idx]} (score={scores[idx]:.2f})")
