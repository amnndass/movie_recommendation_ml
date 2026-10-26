from shutil import move
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


print("loading semantic model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

movies = [
    {
        "title": "Interstellar",
        "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival as Earth faces famine."
    },
    {
        "title": "The Dark Knight",
        "description": "A vigilante superhero battles a chaotic criminal mastermind known as the Joker in a corrupt city."
    },
    {
        "title": "Superbad",
        "description": "Two co-dependent high school seniors navigate peer pressure, parties, and awkward moments before going to college."
    },
    {
        "title": "Spirited Away",
        "description": "A young girl enters a magical world of spirits, witches, and bizarre creatures to rescue her parents."
    },
    {
        "title": "A Quiet Place",
        "description": "A family struggles to survive in a post-apocalyptic world inhabited by blind extraterrestrial creatures with hypersensitive hearing."
    },
    {
        "title": "La La Land",
        "description": "A jazz pianist and an aspiring actress pursue their career dreams and fall in love in vibrant Los Angeles."
    },
    {
        "title": "Inception",
        "description": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into a CEO's mind."
    }
]

print(f"indexing {len(movies)} movies into 384 dimensioal vector space...")
descriptions = [m['description'] for m in movies]
movie_embeddings = model.encode(descriptions)

print(f"index ready, database matrix shape: {movie_embeddings.shape}\n")

def search(query, top_k=2):
    query_vec = model.encode([query])

    scores = cosine_similarity(query_vec, movie_embeddings)[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    print(f"query: \"{query}\"")
    print('=' *60)

    for rank, idx in enumerate(top_indices, 1):
        movie = movies[idx]
        match_pct = scores[idx] * 100
        print(f"{rank} {movie['title']} [match: {match_pct:.1f}%]")
        print(f"   \"{movie['description']}\"\n")

search("creepy silent mosnters that hunt by sound")

def recommend_similar(movie_title, top_k=2):

    target_idx = None
    for i,m in enumerate(movies):
        if m['title'].lower() == movie_title.lower():
            target_idx = i
            break

    if target_idx is None:
        print(f"movie {movie_title} not found in database")
        return

    target_vec = movie_embeddings[target_idx]

    scores = cosine_similarity([target_vec], movie_embeddings)[0]

    similar_indices = np.argsort(scores)[::-1][1:top_k+1]

    print(f"beacuse of watched {movies[target_idx]['title']}")
    print("="*60)

    for rank, idx in enumerate(similar_indices, 1):
        movie = movies[idx]
        match_pct = scores[idx] * 100
        print(f"{rank} {movie['title']} [similarity: {match_pct:.1f}%]")
        print(f"    \"{movie['description']}\"\n")

recommend_similar('Interstellar')
recommend_similar('Superbad')
    
