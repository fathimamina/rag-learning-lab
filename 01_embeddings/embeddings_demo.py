from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Python is a programming language",
    "TensorFlow is a deep learning framework",
    "CNN is used for image processing"
]

embeddings = model.encode(sentences)

print("Shape:", embeddings.shape)
print("\nFirst 10 values of first embedding:")
print(embeddings[0][:10])