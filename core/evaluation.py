import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import streamlit as st

def calculate_ils(recommendations, df_scaled, audio_features):
    """
    Calculates the Intra-List Similarity (ILS) of the recommendations.
    ILS measures the average similarity between all pairs of recommended items.
    A higher ILS means the recommendations are more similar (less diverse).
    A lower ILS means the recommendations are more diverse.
    """
    if len(recommendations) <= 1:
        return 0.0
        
    rec_ids = recommendations["track_id"].values
    rec_scaled = df_scaled[df_scaled["track_id"].isin(rec_ids)]
    
    if len(rec_scaled) <= 1:
        return 0.0
        
    rec_vectors = rec_scaled[audio_features].values
    sim_matrix = cosine_similarity(rec_vectors)
    
    # Exclude self-similarity diagonal
    n = sim_matrix.shape[0]
    tri_upper = np.triu_indices(n, k=1)
    mean_sim = np.mean(sim_matrix[tri_upper])
    
    return float(mean_sim)

def evaluate_holdout_hit_rate(df, df_scaled, audio_features, recommender_func, genre, seeds_count=3, holdout_count=5, top_n=20, same_genre_only=False, **kwargs):
    """
    Evaluates recommendation quality by simulating a hold-out test on a specific genre.
    Takes a set of tracks of a genre, holds out some, uses others as seeds, and measures
    how many hold-out tracks are successfully recommended in the top_n.
    """
    # 1. Get tracks of the selected genre
    genre_tracks = df[df["track_genre"] == genre]["track_id"].values
    if len(genre_tracks) < (seeds_count + holdout_count):
        return {
            "hit_rate": 0.0,
            "hits": 0,
            "total_holdout": 0,
            "error": f"Faixas insuficientes no gênero {genre} para realizar o teste."
        }
        
    # 2. Randomly select seeds and holdouts
    shuffled_tracks = np.random.choice(genre_tracks, size=seeds_count + holdout_count, replace=False)
    seed_ids = shuffled_tracks[:seeds_count]
    holdout_ids = shuffled_tracks[seeds_count:]
    
    # 3. Generate recommendations
    recs = recommender_func(
        seed_ids=list(seed_ids),
        df=df,
        df_scaled=df_scaled,
        audio_features=audio_features,
        top_n=top_n,
        same_genre_only=same_genre_only,
        **kwargs
    )
    
    if recs.empty:
        return {"hit_rate": 0.0, "hits": 0, "total_holdout": holdout_count}
        
    # 4. Calculate Hits
    rec_ids = recs["track_id"].values
    hits = len(set(holdout_ids).intersection(set(rec_ids)))
    hit_rate = hits / holdout_count
    
    return {
        "hit_rate": hit_rate,
        "hits": hits,
        "total_holdout": holdout_count,
        "seed_tracks": list(df[df["track_id"].isin(seed_ids)]["track_name"].values),
        "holdout_tracks": list(df[df["track_id"].isin(holdout_ids)]["track_name"].values),
        "hits_names": list(df[df["track_id"].isin(list(set(holdout_ids).intersection(set(rec_ids))))]["track_name"].values)
    }

def evaluate_genre_precision(df, df_scaled, audio_features, recommender_func,
                             genre, seeds_count=3, top_n=10, repetitions=20, **kwargs):
    """
    Precisão de Gênero@N (proxy de acurácia, sem histórico de interações).

    A partir de `seeds_count` faixas-semente de um único gênero, gera a
    recomendação ABERTA (Top-N sobre todo o catálogo, sem filtro de gênero) e
    mede a fração das recomendações que pertencem ao mesmo gênero da semente.
    O valor final é a média dessa fração ao longo de `repetitions` execuções.
    A referência (acaso) é 1/(nº de gêneros do catálogo).
    """
    genre_tracks = df[df["track_genre"] == genre]["track_id"].values
    if len(genre_tracks) < seeds_count + 1:
        return {"precision": 0.0, "reps": 0, "gain": 0.0, "baseline": 0.0,
                "error": f"Faixas insuficientes no gênero {genre} para o teste."}

    precisions = []
    for _ in range(repetitions):
        seed_ids = list(np.random.choice(genre_tracks, size=seeds_count, replace=False))
        recs = recommender_func(
            seed_ids=seed_ids,
            df=df,
            df_scaled=df_scaled,
            audio_features=audio_features,
            top_n=top_n,
            same_genre_only=False,
            **kwargs
        )
        if recs.empty:
            continue
        precisions.append(float((recs["track_genre"] == genre).mean()))

    mean_p = float(np.mean(precisions)) if precisions else 0.0
    n_genres = int(df["track_genre"].nunique())
    baseline = 1.0 / n_genres if n_genres else 0.0
    return {
        "precision": mean_p,
        "reps": len(precisions),
        "baseline": baseline,
        "gain": (mean_p / baseline) if baseline else 0.0,
        "n_genres": n_genres,
    }

def calculate_catalog_coverage(df, df_scaled, audio_features, recommender_func, n_tests=50, top_n=10, **kwargs):
    """
    Calculates Catalog Coverage.
    Catalog Coverage is the percentage of all tracks in the catalog that are recommended
    at least once over a large number of random recommendation queries.
    """
    recommended_tracks = set()
    all_track_ids = df["track_id"].values
    
    # Run recommendation queries using random seeds
    for _ in range(n_tests):
        # Pick 3 random seed tracks
        seed_ids = list(np.random.choice(all_track_ids, size=3, replace=False))
        recs = recommender_func(
            seed_ids=seed_ids,
            df=df,
            df_scaled=df_scaled,
            audio_features=audio_features,
            top_n=top_n,
            **kwargs
        )
        if not recs.empty:
            recommended_tracks.update(recs["track_id"].values)
            
    coverage = len(recommended_tracks) / len(df)
    return coverage, len(recommended_tracks)

def calculate_kmeans_metrics(df_scaled, audio_features, cluster_labels, sample_size=2000):
    """
    Calculates Silhouette Score for evaluating K-Means clustering quality.
    Uses a random sample of the data to keep it fast.
    """
    if len(df_scaled) <= sample_size:
        sample_df = df_scaled
        sample_labels = cluster_labels
    else:
        np.random.seed(42)
        indices = np.random.choice(len(df_scaled), size=sample_size, replace=False)
        sample_df = df_scaled.iloc[indices]
        sample_labels = cluster_labels[indices]
        
    try:
        score = silhouette_score(sample_df[audio_features], sample_labels)
    except Exception as e:
        score = 0.0
        
    return score
