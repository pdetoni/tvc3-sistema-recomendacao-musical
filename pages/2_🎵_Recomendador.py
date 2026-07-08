import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from core.data_loader import load_and_preprocess_data
from core.recommender import recommend_cosine, recommend_kmeans, train_kmeans
from core.evaluation import calculate_ils

st.set_page_config(
    page_title="Recomendador Musical",
    page_icon="🎵",
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
        background: linear-gradient(135deg, #FF0844 0%, #FFB199 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }
    .seed-card {
        background-color: #1E222B;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #FF0844;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .seed-title {
        font-weight: bold;
        color: #FAFAFA;
        margin: 0;
    }
    .seed-artist {
        font-size: 0.85rem;
        color: #A0AEC0;
        margin: 0;
    }
    .rec-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: #FFB199;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='page-title'>🎵 Motor de Recomendação Musical</h1>", unsafe_allow_html=True)

# Load data
try:
    df, df_scaled, is_mock, audio_features = load_and_preprocess_data()
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

if is_mock:
    st.warning("⚠️ Modo de Demonstração ativo. As recomendações serão baseadas nos dados simulados.")

# Session State Initialization
if "seed_tracks" not in st.session_state:
    st.session_state.seed_tracks = []

# Sidebar configurations
st.sidebar.header("🎛️ Configurações da Recomendação")

rec_method = st.sidebar.selectbox(
    "Algoritmo:",
    ["Similaridade do Cosseno", "Agrupamento K-Means"],
    help="Escolha o método matemático para o cálculo de proximidade das músicas."
)

top_n = st.sidebar.slider(
    "Quantidade de Recomendações (Top-N):",
    min_value=5,
    max_value=30,
    value=10,
    step=5
)

genre_strategy = st.sidebar.radio(
    "Filtro de Gênero:",
    ["Sem restrições (Recomendação aberta)", "Apenas mesmos gêneros das sementes", "Filtrar por gêneros específicos"],
    index=0
)

# Handle genre choices
genre_filter = None
same_genre_only = False

if genre_strategy == "Apenas mesmos gêneros das sementes":
    same_genre_only = True
elif genre_strategy == "Filtrar por gêneros específicos":
    all_genres = sorted(df["track_genre"].unique())
    genre_filter = st.sidebar.multiselect("Selecione os gêneros:", all_genres)

# Additional parameter for K-Means
if rec_method == "Agrupamento K-Means":
    n_clusters = st.sidebar.slider("Quantidade de Clusters (K-Means):", 5, 25, 10, step=1)
    # Train/retrieve KMeans
    kmeans_model, cluster_labels = train_kmeans(df_scaled, audio_features, n_clusters=n_clusters)
else:
    cluster_labels = None

# Interface Layout
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### 🔍 1. Adicione Músicas que Você Gosta")
    st.write("Digite o nome da música ou o artista no campo abaixo e adicione à sua lista de sementes (máx. 5):")
    
    # Search logic
    search_query = st.text_input("Buscar música ou artista:", placeholder="Ex: Coldplay, Yellow, Taylor Swift...")
    
    if search_query:
        # Simple case-insensitive search
        search_mask = df["track_name"].str.contains(search_query, case=False, na=False) | \
                      df["artists"].str.contains(search_query, case=False, na=False)
        search_results = df[search_mask].head(8)
        
        if not search_results.empty:
            for idx, row in search_results.iterrows():
                track_id = row["track_id"]
                track_name = row["track_name"]
                artist = row["artists"]
                
                # Check if already added
                is_seed = track_id in st.session_state.seed_tracks
                
                col_btn_txt, col_btn_act = st.columns([4, 1])
                with col_btn_txt:
                    st.write(f"**{track_name}**  \n*{artist}*")
                with col_btn_act:
                    if is_seed:
                        st.button("✅", key=f"added_{track_id}", disabled=True)
                    else:
                        if st.button("➕", key=f"add_{track_id}"):
                            if len(st.session_state.seed_tracks) >= 5:
                                st.error("Você já atingiu o limite de 5 faixas-semente!")
                            else:
                                st.session_state.seed_tracks.append(track_id)
                                st.rerun()
                st.markdown("<hr style='margin: 0.2rem 0; opacity: 0.3;'>", unsafe_allow_html=True)
        else:
            st.info("Nenhuma música encontrada com essa busca. Tente outras palavras-chave.")
            
    # Selected Seeds display
    st.write("")
    st.markdown("#### 🌱 Suas Faixas-Semente Selecionadas:")
    
    if len(st.session_state.seed_tracks) > 0:
        for seed_id in list(st.session_state.seed_tracks):
            seed_row = df[df["track_id"] == seed_id].iloc[0]
            
            # Seed Card html/Streamlit mix
            col_seed_info, col_seed_del = st.columns([5, 1])
            with col_seed_info:
                st.markdown(f"""
                <div class='seed-card'>
                    <div>
                        <p class='seed-title'>{seed_row['track_name']}</p>
                        <p class='seed-artist'>{seed_row['artists']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_seed_del:
                # Add a vertical spacer to align
                st.write("")
                if st.button("🗑️", key=f"del_{seed_id}"):
                    st.session_state.seed_tracks.remove(seed_id)
                    st.rerun()
                    
        # Clear all seeds
        if st.button("Limpar todas as sementes"):
            st.session_state.seed_tracks = []
            st.rerun()
    else:
        st.info("Sua lista está vazia. Adicione músicas usando o campo de busca acima.")

with col_right:
    st.markdown("### 🎯 2. Recomendações Geradas")
    
    if len(st.session_state.seed_tracks) == 0:
        st.info("💡 Adicione pelo menos uma faixa-semente na coluna à esquerda para gerar recomendações.")
        
        # Give user a quick setup option
        st.write("Ou clique nos botões abaixo para usar sementes de exemplo de diferentes vibes:")
        
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        with col_ex1:
            if st.button("🎸 Vibração Rock"):
                rock_seeds = df[df["track_genre"].isin(["rock", "metal"])].head(3)["track_id"].tolist()
                st.session_state.seed_tracks = rock_seeds
                st.rerun()
        with col_ex2:
            if st.button("🎹 Vibração Jazz/Clássica"):
                jazz_seeds = df[df["track_genre"].isin(["jazz", "classical"])].head(3)["track_id"].tolist()
                st.session_state.seed_tracks = jazz_seeds
                st.rerun()
        with col_ex3:
            if st.button("⚡ Vibração Eletrônica/Pop"):
                pop_seeds = df[df["track_genre"].isin(["electronic", "pop"])].head(3)["track_id"].tolist()
                st.session_state.seed_tracks = pop_seeds
                st.rerun()
    else:
        # Run the recommendations
        with st.spinner("Calculando recomendações..."):
            if rec_method == "Similaridade do Cosseno":
                recommendations = recommend_cosine(
                    seed_ids=st.session_state.seed_tracks,
                    df=df,
                    df_scaled=df_scaled,
                    audio_features=audio_features,
                    top_n=top_n,
                    genre_filter=genre_filter,
                    same_genre_only=same_genre_only
                )
            else: # KMeans
                recommendations = recommend_kmeans(
                    seed_ids=st.session_state.seed_tracks,
                    df=df,
                    df_scaled=df_scaled,
                    audio_features=audio_features,
                    cluster_labels=cluster_labels,
                    top_n=top_n,
                    genre_filter=genre_filter,
                    same_genre_only=same_genre_only
                )
                
        if recommendations.empty:
            st.warning("Não foi possível gerar recomendações com os filtros de gênero selecionados. Tente relaxar as restrições de gênero na barra lateral.")
        else:
            # Stats about recommendations
            ils_value = calculate_ils(recommendations, df_scaled, audio_features)
            
            st.success(f"Recomendações geradas com sucesso usando **{rec_method}**!")
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Mapeamento Gênero", f"{recommendations['track_genre'].nunique()} gênero(s) sugerido(s)")
            with col_stat2:
                # ILS explanation (lower value = more diverse)
                st.metric("Métrica de Diversidade (ILS)", f"{ils_value:.3f}", 
                          help="Intra-List Similarity. Quanto menor o valor, mais diversas são as músicas recomendadas em termos de características de áudio.")
            
            # Show recommendations Table/List
            st.markdown("<p class='rec-header'>Músicas Recomendadas</p>", unsafe_allow_html=True)
            
            # Format and display
            display_recs = recommendations.copy()
            # Convert similarity score to readable percentage
            display_recs["Compatibilidade"] = (display_recs["similarity_score"] * 100).map("{:.1f}%".format)
            
            # Select columns to show nicely
            table_cols = ["track_name", "artists", "album_name", "popularity", "track_genre", "Compatibilidade"]
            display_recs.columns = ["track_id", "Artistas", "Álbum", "Nome", "Popularidade", "duration_ms", "explicit", 
                                    "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", 
                                    "instrumentalness", "liveness", "valence", "tempo", "time_signature", "Gênero", 
                                    "similarity_score"] + (["cluster"] if "cluster" in display_recs.columns else []) + ["Compatibilidade"]
            
            # Let's map back to user columns
            final_display = display_recs[["Nome", "Artistas", "Álbum", "Gênero", "Popularidade", "Compatibilidade"]]
            st.dataframe(final_display, hide_index=True, use_container_width=True)
            
            # Visualize radar chart comparison (Wow factor)
            st.markdown("---")
            st.markdown("### 📊 Perfil de Áudio: Sementes vs Recomendações")
            st.write("Este gráfico radar compara o perfil de áudio médio das suas músicas escolhidas (sementes) com o perfil médio das músicas recomendadas:")
            
            # Get average features
            seed_data = df_scaled[df_scaled["track_id"].isin(st.session_state.seed_tracks)][audio_features].mean().reset_index()
            seed_data.columns = ["Feature", "Valor"]
            seed_data["Grupo"] = "Suas Sementes"
            
            rec_scaled_df = df_scaled[df_scaled["track_id"].isin(recommendations["track_id"])]
            rec_data = rec_scaled_df[audio_features].mean().reset_index()
            rec_data.columns = ["Feature", "Valor"]
            rec_data["Grupo"] = "Recomendações"
            
            # Plotly expects a specific structure or we can build go.Figure
            # To scale features nicely on a radar chart (since we want them between 0 and 1 or standard scaler bounds), 
            # let's rescale the Standard Scaled features back or use MinMax values of original features so they are 0 to 1
            # MinMax is much better for radar chart visual representation!
            # Let's do MinMax scaling of the average original features
            orig_seeds_avg = df[df["track_id"].isin(st.session_state.seed_tracks)][[
                "danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness", "liveness"
            ]].mean()
            
            orig_recs_avg = recommendations[[
                "danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness", "liveness"
            ]].mean()
            
            radar_features = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness", "liveness"]
            radar_labels = ["Dançabilidade", "Energia", "Valência (Alegria)", "Acústica", "Instrumental", "Discurso/Voz", "Vivacidade"]
            
            fig_radar = go.Figure()
            
            # Add Seeds trace
            fig_radar.add_trace(go.Scatterpolar(
                r=orig_seeds_avg.values,
                theta=radar_labels,
                fill='toself',
                name='Suas Sementes (Média)',
                line_color='#FF0844',
                opacity=0.6
            ))
            
            # Add Recommendations trace
            fig_radar.add_trace(go.Scatterpolar(
                r=orig_recs_avg.values,
                theta=radar_labels,
                fill='toself',
                name='Recomendações (Média)',
                line_color='#00C6FF',
                opacity=0.6
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                template="plotly_dark",
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
