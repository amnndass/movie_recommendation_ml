import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Semantic AI", layout='centered')
st.title("AI Movie Search & Recommender")
st.write("Powered by 384 dense neural embeddigs and vector similarity.")

@st.cache_resource
def load_resources():
    df = pd.read_csv("movies.csv")
    embeddings = np.load("movie_embeddings.npy")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, embeddings, model

df, embeddings, model = load_resources()

tab1, tab2 = st.tabs(['Semantic Search', 'Movie Recommender'])

with tab1:
    st.subheader("Describe what you want to watch:")
    query = st.text_input("search by plot, them or mood:", placeholder="eg. space exploration or funny heist")
    if query:
        query_vec = model.encode([query])
        scores = cosine_similarity(query_vec, embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:3]
    
        st.write('top matches:')
        for rank, idx in enumerate(top_indices, 1):
            row = df.iloc[idx]
            match_pct = scores[idx] * 100
            st.markdown(f"{rank} {row['title']}** *({row['director']})* - {match_pct:.1f}% match")
            st.write(row['overview'])
            st.progress(float(scores[idx]))

with tab2:
    st.subheader("pick a movie you love")
    selected_movie = st.selectbox("choose a title:", df['title'].tolist())
    recommend_btn = st.button("find similar movies", type="primary")
    if recommend_btn:
        target_idx = df[df['title'] == selected_movie].index[0]
        scores = cosine_similarity([embeddings[target_idx]], embeddings)[0]
        similar_indices = np.argsort(scores)[::-1][1:4]  # Slices indices 1, 2, 3 (skips self)
        
        st.write(f"### Because you liked *{selected_movie}*:")
        for rank, idx in enumerate(similar_indices, 1):
            row = df.iloc[idx]
            st.markdown(f"**#{rank} {row['title']}** *({row['director']})* — `{scores[idx]*100:.1f}%`")
            st.write(row['overview'])
            st.progress(float(scores[idx]))
