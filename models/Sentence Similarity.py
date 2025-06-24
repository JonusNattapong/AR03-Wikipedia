from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
sentences = ["I love cats.", "I adore cats.", "Dogs are great pets."]

embeddings = model.encode(sentences)
similarity = util.cos_sim(embeddings, embeddings)
print(similarity)
