
from sentence_transformers.base.evaluation import SentenceEvaluator
import pandas as pd

df = pd.read_csv("movies.csv")

print(f"loaded {len(df)} movies")

print(df[['title', 'genre', 'director']].head(2))

def create_feature_soup(row):
    return f"Genre: {row['genre']}. Director: {row['director']}. Plot: {row['overview']}"

df['soup'] = df.apply(create_feature_soup, axis=1)
print("\n sample feature soup for movie #1", df['soup'].iloc[0])

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(df['soup'].tolist(), show_progress_bar=True)


# np.save('movie_embeddings.npy', embeddings)
# print(f"successfully saved embeddings matrix of shape {embeddings.shape} to 'movie_embeddings.npy'")

from sklearn.metrics.pairwise import cosine_similarity

saved_embeddings = np.load("movie_embeddings.npy")

def search(query, top_k=3):
    query_vec = model.encode([query])
    scores = cosine_similarity(query_vec, saved_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return df.iloc[top_indices].assign(similarity=scores[top_indices])

results = search("Christopher nolan psychological space thriller")

for _, row in results.iterrows():
    print(f"{row['title']} ({row['director']}) - Match: {row['similarity']*100:.1f}%")
    print(f"    {row['genre']} | {row['overview']}")

def recommend_similar(movie_title, top_k=3):
    idx = df[df['title'].str.lower() == movie_title.lower()].index[0]
    scores = cosine_similarity([saved_embeddings[idx]], saved_embeddings)[0]

    top_indices = np.argsort(scores)[::-1][1:top_k]
    return df.iloc[top_indices].assign(similarity=scores[top_indices])

print("recommendations fo pulp fiction")
recs = recommend_similar("Pulp Fiction", top_k=3)

for _, row in recs.iterrows():
    print(f"{row['title']} ({row['director']}) = similarity: {row['similarity']*100:.1f}%")
