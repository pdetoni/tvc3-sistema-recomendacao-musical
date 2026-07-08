import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from core.data_loader import load_and_preprocess_data

st.set_page_config(
    page_title="EDA - Explorar Dados",
    page_icon="🔍",
    layout="wide"
)

# Custom css for titles and margins
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .page-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E222B;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        border-top: 3px solid #00C6FF;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: bold;
        color: #FAFAFA;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #A0AEC0;
        text-transform: uppercase;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='page-title'>Análise Exploratória de Dados (EDA)</h1>", unsafe_allow_html=True)

# Load data
try:
    df, df_scaled, is_mock, audio_features = load_and_preprocess_data()
except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
    st.stop()

# System status message
if is_mock:
    st.warning("⚠️ Mostrando dados simulados para demonstração. Carregue o arquivo `dataset.csv` para ver os dados reais.")

# Section 1: Overview Metrics
st.markdown("### 📊 Visão Geral do Catálogo")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{len(df):,}</div><div class='metric-lbl'>Total de Músicas</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{df['artists'].nunique():,}</div><div class='metric-lbl'>Artistas Únicos</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{df['track_genre'].nunique()}</div><div class='metric-lbl'>Gêneros Resgistrados</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{df['popularity'].mean():.1f}</div><div class='metric-lbl'>Popularidade Média</div></div>", unsafe_allow_html=True)

st.write("")

# Tabs for organization
tab1, tab2, tab3 = st.tabs(["🎵 Características de Áudio", "🏷️ Gêneros & Popularidade", "📄 Visualização dos Dados Brutos"])

with tab1:
    st.markdown("### Distribuição das Características de Áudio")
    st.write("Selecione uma característica para visualizar como ela se comporta ao longo das faixas do catálogo:")
    
    selected_feature = st.selectbox("Escolha a característica de áudio:", audio_features, index=0)
    
    col_feat1, col_feat2 = st.columns([2, 1])
    
    with col_feat1:
        # Create a styled histogram
        fig_hist = px.histogram(
            df, 
            x=selected_feature, 
            nbins=40,
            title=f"Histograma de {selected_feature.capitalize()}",
            color_discrete_sequence=["#00C6FF"],
            template="plotly_dark"
        )
        fig_hist.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_feat2:
        # Descriptive statistics for selected feature
        desc = df[selected_feature].describe()
        st.markdown(f"#### 📈 Estatísticas Descritivas: `{selected_feature}`")
        stats_df = pd.DataFrame({
            "Métrica": ["Média", "Desvio Padrão", "Mínimo", "25% (Q1)", "Mediana (Q2)", "75% (Q3)", "Máximo"],
            "Valor": [f"{desc['mean']:.4f}", f"{desc['std']:.4f}", f"{desc['min']:.4f}", f"{desc['25%']:.4f}", f"{desc['50%']:.4f}", f"{desc['75%']:.4f}", f"{desc['max']:.4f}"]
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
    st.markdown("---")
    st.markdown("### Matriz de Correlação das Características")
    st.write("A matriz de correlação nos ajuda a identificar como as características de áudio se relacionam. Por exemplo, músicas com alta *energia* tendem a ser mais barulhentas (*loudness*) e menos *acústicas*.")
    
    # Compute correlation
    corr = df[audio_features].corr()
    
    # Create correlation heatmap
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        zmin=-1, zmax=1,
        title="Correlação de Pearson entre Features de Áudio",
        template="plotly_dark"
    )
    fig_corr.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    st.markdown("### Gêneros Musicais e Popularidade")
    
    col_gen1, col_gen2 = st.columns(2)
    
    with col_gen1:
        st.markdown("#### Top 15 Gêneros com Mais Músicas")
        top_genres = df["track_genre"].value_counts().head(15).reset_index()
        top_genres.columns = ["Gênero", "Quantidade de Faixas"]
        
        fig_genre = px.bar(
            top_genres,
            x="Quantidade de Faixas",
            y="Gênero",
            orientation="h",
            color="Quantidade de Faixas",
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        fig_genre.update_layout(
            yaxis={"categoryorder":"total ascending"},
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_genre, use_container_width=True)
        
    with col_gen2:
        st.markdown("#### Distribuição de Popularidade")
        fig_pop = px.histogram(
            df,
            x="popularity",
            nbins=30,
            title="Histograma de Popularidade das Faixas",
            color_discrete_sequence=["#FF4B4B"],
            template="plotly_dark"
        )
        fig_pop.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_pop, use_container_width=True)
        
    st.markdown("---")
    st.markdown("### 🪐 Espaço Bi-dimensional de Áudio")
    st.write("Veja como as faixas se posicionam no espaço bi-dimensional. Selecione duas características e um subconjunto de gêneros para explorar a dispersão das músicas:")
    
    # Selection options for scatter
    scatter_feat_x = st.selectbox("Eixo X:", audio_features, index=0)
    scatter_feat_y = st.selectbox("Eixo Y:", audio_features, index=1)
    
    all_genres = sorted(df["track_genre"].unique())
    selected_scatter_genres = st.multiselect(
        "Filtrar por gêneros (máx 5 recomendados para legibilidade):",
        all_genres,
        default=all_genres[:4] if len(all_genres) > 4 else all_genres
    )
    
    if len(selected_scatter_genres) > 0:
        scatter_df = df[df["track_genre"].isin(selected_scatter_genres)]
        # Limit points to plot for performance
        if len(scatter_df) > 3000:
            scatter_df = scatter_df.sample(3000, random_state=42)
            
        fig_scatter = px.scatter(
            scatter_df,
            x=scatter_feat_x,
            y=scatter_feat_y,
            color="track_genre",
            hover_name="track_name",
            hover_data=["artists", "album_name"],
            title=f"Dispersão: {scatter_feat_x.capitalize()} vs {scatter_feat_y.capitalize()}",
            template="plotly_dark",
            opacity=0.7
        )
        fig_scatter.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Por favor, selecione pelo menos um gênero para gerar o gráfico de dispersão.")

with tab3:
    st.markdown("### Visualização dos Dados Brutos")
    st.write("Visualize e explore amostras do catálogo completo:")
    
    n_rows = st.slider("Número de linhas para exibir:", 5, 100, 10)
    
    # Columns to show
    cols_to_show = ["track_name", "artists", "album_name", "popularity", "track_genre"] + audio_features
    st.dataframe(df[cols_to_show].sample(n_rows, random_state=42), use_container_width=True)
