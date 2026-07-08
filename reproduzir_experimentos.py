#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduzir_experimentos.py
==========================

Script único e determinístico que regenera todas as tabelas da seção de
Resultados do relatório (docs/relatorio.tex) a partir da base bruta.

Uso:
    1. Baixe o Spotify Tracks Dataset (Kaggle, maharshipandya) e salve como
       data/dataset.csv
    2. Na raiz do projeto, execute:
           python reproduzir_experimentos.py

O script é autocontido (não depende do Streamlit): ele replica exatamente o
mesmo pré-processamento de core/data_loader.py e a mesma lógica de
recomendação/avaliação de core/recommender.py e core/evaluation.py, fixando a
semente aleatória em 42 em todas as etapas estocásticas. Assim, execuções
repetidas produzem exatamente os mesmos números impressos abaixo.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
DATA_PATH = os.path.join("data", "dataset.csv")

# Mesmas 11 características de áudio de core/data_loader.py
AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

# Gêneros avaliados na Precisão de Gênero (chave no dataset -> rótulo no relatório)
GENRES_AVALIADOS = [
    ("classical", "Clássica"),
    ("rock", "Rock"),
    ("pop", "Pop"),
    ("metal", "Metal"),
    ("jazz", "Jazz"),
]


# ----------------------------------------------------------------------
# 1. Carga e pré-processamento (espelha core/data_loader.py)
# ----------------------------------------------------------------------
def carregar_dados():
    if not os.path.exists(DATA_PATH):
        sys.exit(
            f"ERRO: '{DATA_PATH}' não encontrado.\n"
            "Baixe o Spotify Tracks Dataset no Kaggle e salve como data/dataset.csv.\n"
            "Link: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset"
        )

    df = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.dropna(subset=["track_id", "track_name", "artists"])
    df = df.drop_duplicates(subset=["track_id"]).reset_index(drop=True)
    df["artists"] = df["artists"].astype(str).str.replace(";;", ", ")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[AUDIO_FEATURES])
    df_scaled = pd.DataFrame(scaled, columns=AUDIO_FEATURES)
    df_scaled["track_id"] = df["track_id"]
    df_scaled["track_genre"] = df["track_genre"]
    return df, df_scaled


# ----------------------------------------------------------------------
# 2. Recomendadores (espelham core/recommender.py)
# ----------------------------------------------------------------------
def recommend_cosine(seed_ids, df, df_scaled, top_n=10):
    seed_idx = df_scaled[df_scaled["track_id"].isin(seed_ids)].index
    if len(seed_idx) == 0:
        return pd.DataFrame()
    profile = np.mean(df_scaled.loc[seed_idx, AUDIO_FEATURES].values, axis=0).reshape(1, -1)
    mask = ~df_scaled["track_id"].isin(seed_ids)
    cand = df_scaled[mask]
    sims = cosine_similarity(profile, cand[AUDIO_FEATURES].values).flatten()
    res = df.loc[cand.index].copy()
    res["similarity_score"] = sims
    return res.sort_values("similarity_score", ascending=False).head(top_n)


def recommend_kmeans(seed_ids, df, df_scaled, cluster_labels, top_n=10):
    seed_idx = df_scaled[df_scaled["track_id"].isin(seed_ids)].index
    if len(seed_idx) == 0:
        return pd.DataFrame()
    seed_clusters = np.unique(cluster_labels[seed_idx])
    mask = np.isin(cluster_labels, seed_clusters) & (~df_scaled["track_id"].isin(seed_ids)).values
    cand = df_scaled[mask]
    if len(cand) == 0:
        return pd.DataFrame()
    profile = np.mean(df_scaled.loc[seed_idx, AUDIO_FEATURES].values, axis=0).reshape(1, -1)
    sims = cosine_similarity(profile, cand[AUDIO_FEATURES].values).flatten()
    res = df.loc[cand.index].copy()
    res["similarity_score"] = sims
    return res.sort_values("similarity_score", ascending=False).head(top_n)


# ----------------------------------------------------------------------
# 3. Métricas (espelham core/evaluation.py)
# ----------------------------------------------------------------------
def intra_list_similarity(recs, df_scaled):
    ids = recs["track_id"].values
    vecs = df_scaled[df_scaled["track_id"].isin(ids)][AUDIO_FEATURES].values
    if len(vecs) <= 1:
        return 0.0
    sim = cosine_similarity(vecs)
    iu = np.triu_indices(sim.shape[0], k=1)
    return float(np.mean(sim[iu]))


def genre_precision(df, df_scaled, genre, rec_fn, seeds_count=3, top_n=10,
                    repetitions=20, **kw):
    tracks = df[df["track_genre"] == genre]["track_id"].values
    if len(tracks) < seeds_count + 1:
        return None
    fracs = []
    for _ in range(repetitions):
        seeds = list(np.random.choice(tracks, size=seeds_count, replace=False))
        recs = rec_fn(seeds, df, df_scaled, top_n=top_n, **kw)
        if recs.empty:
            continue
        fracs.append(float((recs["track_genre"] == genre).mean()))
    return float(np.mean(fracs)) if fracs else 0.0


def catalog_coverage(df, df_scaled, rec_fn, n_tests=40, top_n=10, **kw):
    seen = set()
    all_ids = df["track_id"].values
    for _ in range(n_tests):
        seeds = list(np.random.choice(all_ids, size=3, replace=False))
        recs = rec_fn(seeds, df, df_scaled, top_n=top_n, **kw)
        if not recs.empty:
            seen.update(recs["track_id"].values)
    return len(seen) / len(df), len(seen)


def mean_ils(df, df_scaled, rec_fn, n_tests=40, top_n=10, **kw):
    all_ids = df["track_id"].values
    vals = []
    for _ in range(n_tests):
        seeds = list(np.random.choice(all_ids, size=3, replace=False))
        recs = rec_fn(seeds, df, df_scaled, top_n=top_n, **kw)
        if not recs.empty:
            vals.append(intra_list_similarity(recs, df_scaled))
    return float(np.mean(vals)) if vals else 0.0


def silhouette_for_k(df_scaled, k, sample_size=2000):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(df_scaled[AUDIO_FEATURES])
    if len(df_scaled) > sample_size:
        rs = np.random.RandomState(SEED)
        idx = rs.choice(len(df_scaled), size=sample_size, replace=False)
        return float(silhouette_score(df_scaled.iloc[idx][AUDIO_FEATURES], labels[idx]))
    return float(silhouette_score(df_scaled[AUDIO_FEATURES], labels))


# ----------------------------------------------------------------------
# 4. Execução dos experimentos
# ----------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    print("Carregando e pré-processando data/dataset.csv ...")
    df, df_scaled = carregar_dados()
    n_genres = df["track_genre"].nunique()
    print(f"Catálogo efetivo: {len(df):,} faixas únicas | {n_genres} gêneros\n")

    # K-Means de referência (K=10) usado nas tabelas comparativas
    km = KMeans(n_clusters=10, random_state=SEED, n_init=10)
    labels_k10 = km.fit_predict(df_scaled[AUDIO_FEATURES])

    baseline = 1.0 / n_genres

    # ---- Tabela: Precisão de Gênero@10 ----
    print("=" * 62)
    print("PRECISÃO DE GÊNERO@10 (3 sementes, 20 repetições, recomendação aberta)")
    print(f"Referência (acaso) = 1/{n_genres} = {baseline*100:.2f}%")
    print("-" * 62)
    print(f"{'Gênero':<12}{'Precisão@10':>14}{'Ganho vs. acaso':>18}")
    for key, label in GENRES_AVALIADOS:
        if key not in set(df["track_genre"].unique()):
            print(f"{label:<12}{'(ausente)':>14}")
            continue
        np.random.seed(SEED)
        p = genre_precision(df, df_scaled, key, recommend_cosine,
                            seeds_count=3, top_n=10, repetitions=20)
        print(f"{label:<12}{p*100:>13.1f}%{p/baseline:>16.1f}x")
    print("(valores idênticos para Cosseno e K-Means; ver relatório)\n")

    # ---- Tabela: ILS e Cobertura (40 consultas, Top-10) ----
    print("=" * 62)
    print("DIVERSIDADE (ILS) E COBERTURA — 40 consultas aleatórias, Top-10")
    print("-" * 62)
    np.random.seed(SEED)
    ils_cos = mean_ils(df, df_scaled, recommend_cosine, n_tests=40, top_n=10)
    np.random.seed(SEED)
    cov_cos, cnt_cos = catalog_coverage(df, df_scaled, recommend_cosine, n_tests=40, top_n=10)
    np.random.seed(SEED)
    ils_km = mean_ils(df, df_scaled, recommend_kmeans, n_tests=40, top_n=10,
                      cluster_labels=labels_k10)
    np.random.seed(SEED)
    cov_km, cnt_km = catalog_coverage(df, df_scaled, recommend_kmeans, n_tests=40, top_n=10,
                                      cluster_labels=labels_k10)
    print(f"{'Algoritmo':<26}{'Média ILS':>12}{'Cobertura':>22}")
    print(f"{'Similaridade do Cosseno':<26}{ils_cos:>12.3f}{cov_cos*100:>15.3f}% ({cnt_cos})")
    print(f"{'K-Means (K=10)':<26}{ils_km:>12.3f}{cov_km*100:>15.3f}% ({cnt_km})\n")

    # ---- Tabela: Cobertura vs. nº de consultas (Cosseno, Top-10) ----
    print("=" * 62)
    print("COBERTURA vs. Nº DE CONSULTAS (Cosseno, Top-10)")
    print("-" * 62)
    print(f"{'Consultas':<12}{'Cobertura':>12}{'Faixas únicas':>18}")
    for nq in (40, 200, 500):
        np.random.seed(SEED)
        cov, cnt = catalog_coverage(df, df_scaled, recommend_cosine, n_tests=nq, top_n=10)
        print(f"{nq:<12}{cov*100:>11.3f}%{cnt:>18,}")
    print()

    # ---- Tabela: Silhueta em função de K ----
    print("=" * 62)
    print("COEFICIENTE DE SILHUETA em função de K (amostra de 2.000 faixas)")
    print("-" * 62)
    print(f"{'K':<8}{'Silhueta':>12}")
    for k in (5, 8, 10, 12, 15, 20):
        s = silhouette_for_k(df_scaled, k, sample_size=2000)
        print(f"{k:<8}{s:>12.3f}")
    print()
    print("Concluído. Reconcilie estes valores com as tabelas de docs/relatorio.tex.")


if __name__ == "__main__":
    main()
