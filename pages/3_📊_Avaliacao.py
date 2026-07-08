import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from core.data_loader import load_and_preprocess_data
from core.recommender import recommend_cosine, recommend_kmeans, train_kmeans
from core.evaluation import (
    calculate_ils,
    evaluate_genre_precision,
    calculate_catalog_coverage,
    calculate_kmeans_metrics
)

st.set_page_config(
    page_title="Avaliação do Sistema",
    page_icon="📊",
    layout="wide"
)

# Custom styling for premium UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .page-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }
    .eval-card {
        background-color: #1E222B;
        border-radius: 8px;
        padding: 1.5rem;
        border-top: 3px solid #11998e;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .eval-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #FAFAFA;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='page-title'>📊 Métricas e Avaliação de SAD</h1>", unsafe_allow_html=True)

# Load data
try:
    df, df_scaled, is_mock, audio_features = load_and_preprocess_data()
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

if is_mock:
    st.warning("⚠️ Modo de Demonstração ativo. Testes de avaliação rodando com dados simulados.")

# Intro
st.markdown("""
Sistemas de Apoio à Decisão (SAD) baseados em recomendação precisam ser avaliados sob diferentes perspectivas além da precisão direta.
Nesta página, você pode executar simulações e experimentos analíticos para medir a qualidade, diversidade e cobertura das recomendações.
""")

tab_precision, tab_coverage, tab_kmeans = st.tabs([
    "🎯 Precisão de Gênero (Genre Precision@10)",
    "🌐 Cobertura do Catálogo (Coverage)",
    "🧼 Avaliação dos Clusters (K-Means)"
])

# ----------------- TAB 1: GENRE PRECISION -----------------
with tab_precision:
    st.markdown("### Precisão de Gênero (Genre Precision@N)")
    st.write("""
    **Como funciona:** como a base é de nível de faixa (sem histórico de interações usuário-item),
    adotamos a **Precisão de Gênero** como *proxy* de acurácia.
    1. Selecionamos aleatoriamente algumas faixas-semente de **um único gênero**.
    2. Geramos a recomendação **aberta** (Top-N sobre todo o catálogo, sem filtro de gênero).
    3. Medimos a **fração** das recomendações que pertencem ao mesmo gênero da semente.
    4. Repetimos várias vezes e tiramos a **média**, comparando com o acaso (1/nº de gêneros).
    """)

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        eval_genre = st.selectbox(
            "Selecione um Gênero para Testar:",
            sorted(df["track_genre"].unique()),
            index=0
        )
        top_k = st.slider("Top-N recomendações a gerar:", 5, 30, 10, step=5)

    with col_ctrl2:
        seeds_cnt = st.slider("Quantidade de Sementes:", 1, 5, 3)
        reps = st.slider("Repetições (média):", 5, 40, 20, step=5)

    if st.button("Executar Precisão de Gênero", type="primary"):
        # Semente fixa para reprodutibilidade determinística
        np.random.seed(42)
        res_cosine = evaluate_genre_precision(
            df=df,
            df_scaled=df_scaled,
            audio_features=audio_features,
            recommender_func=recommend_cosine,
            genre=eval_genre,
            seeds_count=seeds_cnt,
            top_n=top_k,
            repetitions=reps,
        )

        np.random.seed(42)
        kmeans_model, cluster_labels = train_kmeans(df_scaled, audio_features, n_clusters=10)
        res_kmeans = evaluate_genre_precision(
            df=df,
            df_scaled=df_scaled,
            audio_features=audio_features,
            recommender_func=recommend_kmeans,
            genre=eval_genre,
            seeds_count=seeds_cnt,
            top_n=top_k,
            repetitions=reps,
            cluster_labels=cluster_labels,
        )

        if "error" in res_cosine:
            st.error(res_cosine["error"])
        else:
            baseline = res_cosine["baseline"]
            st.markdown("---")
            st.markdown("#### 📈 Resultados Comparativos")
            st.caption(f"Referência (acaso) = 1/{res_cosine['n_genres']} = {baseline*100:.2f}%  ·  "
                       f"gênero **{eval_genre}**, {seeds_cnt} sementes, Top-{top_k}, {reps} repetições.")

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric(
                    "Precisão — Similaridade do Cosseno",
                    f"{res_cosine['precision']*100:.1f}%",
                    delta=f"{res_cosine['gain']:.1f}× vs. acaso",
                )
            with col_res2:
                st.metric(
                    "Precisão — K-Means Clustering",
                    f"{res_kmeans['precision']*100:.1f}%",
                    delta=f"{res_kmeans['gain']:.1f}× vs. acaso",
                )

            st.info(
                "💡 **Insight:** gêneros acusticamente distintivos (ex.: *classical*) atingem precisão "
                "muito acima do acaso, enquanto gêneros no meio do espectro (ex.: *jazz*, *metal*) caem "
                "ao nível do acaso — evidência de que os gêneros finos do dataset se sobrepõem no espaço de áudio. "
                "Cosseno e K-Means tendem a coincidir, pois o cluster majoritário das sementes é amplo."
            )

# ----------------- TAB 2: CATALOG COVERAGE -----------------
with tab_coverage:
    st.markdown("### Cobertura do Catálogo (Catalog Coverage)")
    st.write("""
    **O que é:**
    A cobertura do catálogo indica a proporção de músicas da base de dados que têm a chance de serem recomendadas. 
    Se um sistema recomenda sempre as mesmas 50 músicas populares, a cobertura é baixíssima (efeito bolha). 
    Sistemas de recomendação de cauda longa saudáveis possuem maior cobertura.
    
    *Nota: Esta simulação executa múltiplas consultas de recomendação fictícias com sementes aleatórias para ver a variedade de músicas recomendadas.*
    """)
    
    n_queries = st.slider("Número de consultas fictícias para simular:", 20, 100, 40, step=10)
    
    if st.button("Calcular Cobertura"):
        with st.spinner("Executando simulação de cobertura..."):
            coverage_cos, count_cos = calculate_catalog_coverage(
                df=df,
                df_scaled=df_scaled,
                audio_features=audio_features,
                recommender_func=recommend_cosine,
                n_tests=n_queries,
                top_n=10
            )
            
            # Train standard KMeans for test
            kmeans_model, cluster_labels = train_kmeans(df_scaled, audio_features, n_clusters=10)
            
            coverage_km, count_km = calculate_catalog_coverage(
                df=df,
                df_scaled=df_scaled,
                audio_features=audio_features,
                recommender_func=recommend_kmeans,
                n_tests=n_queries,
                top_n=10,
                cluster_labels=cluster_labels
            )
            
        st.markdown("---")
        st.markdown("#### 📊 Resultado da Simulação")
        
        col_cov1, col_cov2 = st.columns(2)
        with col_cov1:
            st.metric("Cobertura — Similaridade do Cosseno", f"{coverage_cos*100:.3f}%", 
                      help=f"Recomendou {count_cos} músicas distintas de {len(df)} totais.")
            st.write(f"Das {len(df):,} músicas do catálogo, **{count_cos}** foram sugeridas pelo menos uma vez.")
            
        with col_cov2:
            st.metric("Cobertura — K-Means Clustering", f"{coverage_km*100:.3f}%",
                      help=f"Recomendou {count_km} músicas distintas de {len(df)} totais.")
            st.write(f"Das {len(df):,} músicas do catálogo, **{count_km}** foram sugeridas pelo menos uma vez.")
            
        st.info("💡 **Insight:** na escala do catálogo completo, Cosseno e K-Means apresentam cobertura praticamente equivalente. Como o cluster majoritário das sementes é muito grande, reordenar seus candidatos pela mesma similaridade do cosseno reproduz, na prática, a busca aberta. A cobertura baixa em poucas consultas reflete subamostragem (cauda longa), não efeito-bolha: ela cresce quase linearmente com o número de consultas.")

# ----------------- TAB 3: K-MEANS EVALUATION -----------------
with tab_kmeans:
    st.markdown("### Métricas de Qualidade dos Clusters K-Means")
    st.write("""
    **O que é:**
    Como avaliamos se os grupos (clusters) criados pelo K-Means fazem sentido matematicamente?
    Usamos o **Coeficiente de Silhueta (Silhouette Score)**. Ele varia de -1 a 1:
    - Próximo a **1**: Os clusters estão bem separados e as músicas dentro do mesmo grupo são muito parecidas.
    - Próximo a **0**: Clusters se sobrepõem (separação fraca).
    - Próximo a **-1**: Músicas foram atribuídas ao grupo errado.
    
    *Cálculo executado em uma amostra de 2.000 faixas para fins de performance.*
    """)
    
    k_val = st.slider("Escolha o valor de K (Clusters):", 5, 25, 10, step=1, key="k_val_tab3")
    
    if st.button("Calcular Silhueta"):
        with st.spinner("Treinando K-Means e calculando coeficiente de silhueta..."):
            kmeans_model, cluster_labels = train_kmeans(df_scaled, audio_features, n_clusters=k_val)
            score = calculate_kmeans_metrics(df_scaled, audio_features, cluster_labels, sample_size=2000)
            
        st.markdown("---")
        st.markdown("#### 🔬 Resultado")
        
        st.metric("Coeficiente de Silhueta Médio", f"{score:.4f}")
        
        # Display feedback based on score
        if score > 0.3:
            st.success("Ótimo coeficiente! Os clusters de música estão bem definidos no espaço de áudio.")
        elif score > 0.1:
            st.info("Coeficiente aceitável. Há alguma sobreposição natural entre estilos musicais vizinhos, o que é esperado no domínio de música.")
        else:
            st.warning("Coeficiente baixo. Os agrupamentos possuem muita interseção de características.")
            
        # Draw a distribution chart of tracks per cluster
        st.write("")
        st.markdown("##### Distribuição de Músicas por Cluster:")
        counts_df = pd.DataFrame({
            "Cluster": [f"Cluster {i}" for i in range(k_val)],
            "Quantidade de Faixas": np.bincount(cluster_labels, minlength=k_val)
        })
        
        fig_dist = px.bar(
            counts_df,
            x="Cluster",
            y="Quantidade de Faixas",
            color="Quantidade de Faixas",
            color_continuous_scale="Cividis",
            template="plotly_dark"
        )
        fig_dist.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
