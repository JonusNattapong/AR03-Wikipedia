from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
query = "How to cook pasta?"
candidates = ["Cook pasta with sauce.", "Make coffee.", "Bake a cake."]

scores = model.predict([[query, c] for c in candidates])
ranked = sorted(
    zip(candidates, scores),
    key=lambda x: x[1],
    reverse=True
)
print(ranked)
