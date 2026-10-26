from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("loading neural model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    'this moview was fantastic and thrilling',
    'the film was amazing and exciting',
    'the phonep battery died in two hours',
    'it was not bad at all, actually quite enjoyable'
]

print("encoding sentences into vectors")
embeddings = model.encode(sentences)

print(f"shape of embeddings matrix: {embeddings.shape}")
print(f"each sentence is now represented by: {embeddings.shape[1]} continous numbers")

for i in range(len(sentences)):
    sim = cosine_similarity([embeddings[0]], [embeddings[i]])[0][0]
    print(f"similiarity with sentence 0")
    print(f"-> \"{sentences[i]}\"")
    print(f"score: {sim:.4f}\n")