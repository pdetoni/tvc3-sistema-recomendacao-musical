import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import streamlit as st

@st.cache_resource
def train_kmeans(df_scaled, audio_features, n_clusters=10):
    """
    Fits a K-Means model to the scaled audio features.
    Cached as a resource to avoid re-training on every user interaction.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_scaled[audio_features])
    return kmeans, cluster_labels

def recommend_cosine(seed_ids, df, df_scaled, audio_features, top_n=10, genre_filter=None, same_genre_only=False):
    """
    Recommends tracks based on Cosine Similarity of audio features.
    
    Parameters:
    - seed_ids: List of track_ids representing user preferences.
    - df: Original DataFrame containing track metadata.
    - df_scaled: DataFrame containing scaled audio features and metadata identifiers.
    - audio_features: List of column names used as audio features.
    - top_n: Number of tracks to recommend.
    - genre_filter: Optional list of genres to restrict the recommendations.
    - same_genre_only: If True, restricts recommendations to the genres present in the seed tracks.
    """
    # 1. Get scaled vectors for the seed tracks
    seed_indices = df_scaled[df_scaled["track_id"].isin(seed_ids)].index
    if len(seed_indices) == 0:
        return pd.DataFrame()
        
    seed_vectors = df_scaled.loc[seed_indices, audio_features].values
    
    # 2. Compute the User Profile Vector (centroid of seeds)
    user_profile = np.mean(seed_vectors, axis=0).reshape(1, -1)
    
    # 3. Get the candidate pool (excluding seed tracks)
    candidate_mask = ~df_scaled["track_id"].isin(seed_ids)
    
    # Apply genre filters if specified
    if same_genre_only:
        seed_genres = df.loc[df["track_id"].isin(seed_ids), "track_genre"].unique()
        candidate_mask = candidate_mask & df_scaled["track_genre"].isin(seed_genres)
    elif genre_filter:
        candidate_mask = candidate_mask & df_scaled["track_genre"].isin(genre_filter)
        
    candidates_scaled = df_scaled[candidate_mask]
    
    if len(candidates_scaled) == 0:
        return pd.DataFrame()
        
    # 4. Compute cosine similarity between User Profile and all candidates
    similarities = cosine_similarity(user_profile, candidates_scaled[audio_features].values).flatten()
    
    # 5. Extract metadata and attach similarity score
    results = df.loc[candidates_scaled.index].copy()
    results["similarity_score"] = similarities
    
    # 6. Sort and select top_n
    recommendations = results.sort_values(by="similarity_score", ascending=False).head(top_n)
    
    return recommendations

def recommend_kmeans(seed_ids, df, df_scaled, audio_features, cluster_labels, top_n=10, genre_filter=None, same_genre_only=False):
    """
    Recommends tracks based on K-Means Clustering.
    Identifies the cluster(s) of the seeds and recommends from those clusters.
    
    Parameters:
    - seed_ids: List of track_ids representing user preferences.
    - df: Original DataFrame containing track metadata.
    - df_scaled: DataFrame containing scaled audio features and metadata identifiers.
    - audio_features: List of column names used as audio features.
    - cluster_labels: Array of cluster assignments for each track in df/df_scaled.
    - top_n: Number of tracks to recommend.
    - genre_filter: Optional list of genres to restrict the recommendations.
    - same_genre_only: If True, restricts recommendations to the genres present in the seed tracks.
    """
    # 1. Identify index/indices of seed tracks
    seed_df = df_scaled[df_scaled["track_id"].isin(seed_ids)]
    seed_indices = seed_df.index
    if len(seed_indices) == 0:
        return pd.DataFrame()
        
    # 2. Get the clusters corresponding to the seed tracks (majority voting or list)
    seed_clusters = cluster_labels[seed_indices]
    unique_clusters, counts = np.unique(seed_clusters, return_counts=True)
    
    # We will draw candidates from the clusters containing our seeds
    # If multiple clusters, we look at tracks in all seed clusters, but prioritize the most frequent one
    candidate_mask = np.isin(cluster_labels, unique_clusters) & (~df_scaled["track_id"].isin(seed_ids))
    
    # Apply genre filters if specified
    if same_genre_only:
        seed_genres = df.loc[df["track_id"].isin(seed_ids), "track_genre"].unique()
        candidate_mask = candidate_mask & df_scaled["track_genre"].isin(seed_genres)
    elif genre_filter:
        candidate_mask = candidate_mask & df_scaled["track_genre"].isin(genre_filter)
        
    candidates_scaled = df_scaled[candidate_mask]
    
    if len(candidates_scaled) == 0:
        # Fallback to no cluster restriction if empty (e.g. genre filter is too restrictive)
        candidate_mask = ~df_scaled["track_id"].isin(seed_ids)
        if same_genre_only:
            seed_genres = df.loc[df["track_id"].isin(seed_ids), "track_genre"].unique()
            candidate_mask = candidate_mask & df_scaled["track_genre"].isin(seed_genres)
        elif genre_filter:
            candidate_mask = candidate_mask & df_scaled["track_genre"].isin(genre_filter)
        candidates_scaled = df_scaled[candidate_mask]
        
    if len(candidates_scaled) == 0:
        return pd.DataFrame()
        
    # 3. Rank the candidates by cosine similarity to the seed centroid (within the cluster)
    seed_vectors = df_scaled.loc[seed_indices, audio_features].values
    user_profile = np.mean(seed_vectors, axis=0).reshape(1, -1)
    
    similarities = cosine_similarity(user_profile, candidates_scaled[audio_features].values).flatten()
    
    results = df.loc[candidates_scaled.index].copy()
    results["similarity_score"] = similarities
    results["cluster"] = cluster_labels[candidates_scaled.index]
    
    # 4. Sort and select top_n
    recommendations = results.sort_values(by="similarity_score", ascending=False).head(top_n)
    
    return recommendations
