# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle



st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("🎬 Movie Recommendation System")
st.markdown("Built using **Collaborative Filtering (User & Item Based)**")


@st.cache_data
def load_data():
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    train = pd.read_csv('Movie Recommendation/ml-100k/u1.base', sep='\t', names=columns)
    movies = pd.read_csv('Movie Recommendation/ml-100k/u.item', sep='|', encoding='latin-1', header=None,
                         names=['movie_id', 'title', 'release_date', 'video_release_date',
                                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                                'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    return train, movies

@st.cache_resource
def load_models():
    with open('Movie Recommendation/user_similarity.pkl', 'rb') as f:
        user_similarity = pickle.load(f)
    with open('Movie Recommendation/item_similarity.pkl', 'rb') as f:
        item_similarity = pickle.load(f)
    return user_similarity, item_similarity

train, movies = load_data()
user_similarity, item_similarity = load_models()


rating_matrix = train.pivot(index='user_id', columns='item_id', values='rating')


def get_top_k_similar(similarity_matrix, target_id, k=5):
    sims = similarity_matrix[target_id].drop(target_id, errors='ignore')
    return sims.nlargest(k)

def predict_ratings_user(user_id, k=5):
    similar_users = get_top_k_similar(user_similarity, user_id, k)
    user_ratings = rating_matrix.loc[user_id]
    weighted_ratings = rating_matrix.T[similar_users.index].dot(similar_users)
    normalization = similar_users.sum()
    preds = weighted_ratings / normalization
    unseen = user_ratings[user_ratings.isna()].index
    return preds.loc[unseen].dropna().sort_values(ascending=False)

def predict_ratings_item(user_id, k=5):
    user_ratings = rating_matrix.loc[user_id].dropna()
    
    st.write("Common items between user ratings and similarity matrix:",
             len(user_ratings.index.intersection(item_similarity.index)))
    
    # Align item IDs between similarity matrix and rating matrix
    common_items = user_ratings.index.intersection(item_similarity.index)
    if len(common_items) == 0:
        return pd.Series([], dtype=float)  # No overlap
    
    # Use only the relevant subset of item similarity
    sims = item_similarity.loc[:, common_items]
    
    # Weighted sum of similarities × user ratings
    weighted_sum = sims.dot(user_ratings.loc[common_items])
    normalization = sims.abs().sum(axis=1)
    preds = weighted_sum / normalization
    
    # Filter unseen movies
    unseen_items = rating_matrix.loc[user_id][rating_matrix.loc[user_id].isna()].index
    preds = preds.loc[preds.index.intersection(unseen_items)]
    
    preds = preds.dropna().sort_values(ascending=False)
    st.write(f"Predictions available for {len(preds)} unseen items.")
    
    return preds




st.sidebar.header("🔧 Controls")
method = st.sidebar.selectbox("Select Recommendation Method", ["User-based", "Item-based"])
user_id = st.sidebar.number_input("Enter User ID", min_value=1, max_value=int(train['user_id'].max()), step=1, value=1)
top_k = st.sidebar.slider("Number of Neighbors (k)", 1, 50, 5)
num_recs = st.sidebar.slider("Number of Recommendations to Display", 5, 20, 10)


if st.sidebar.button("Generate Recommendations"):
    if method == "User-based":
        st.subheader(f"👥 User-based Recommendations for User {user_id}")
        recommendations = predict_ratings_user(user_id, k=top_k).head(num_recs)
    else:
        st.subheader(f"🎞️ Item-based Recommendations for User {user_id}")
        recommendations = predict_ratings_item(user_id, k=top_k).head(num_recs)
    
    rec_movies = movies[movies['movie_id'].isin(recommendations.index)][['movie_id', 'title']]
    rec_movies['Predicted Rating'] = recommendations.values
    rec_movies = rec_movies.sort_values(by='Predicted Rating', ascending=False).reset_index(drop=True)
    st.dataframe(rec_movies, use_container_width=True)

