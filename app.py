import streamlit as st
import os
from core.data_loader import load_and_preprocess_data

# Page configuration
st.set_page_config(
    page_title="SAD: Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #8A2387 50%, #E94057 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #A0AEC0;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: #1E222B;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .card-purple {
        background-color: #1E222B;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #8A2387;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        color: #fff;
    }
    
    .badge-real {
        background-color: #38A169;
    }
    
    .badge-mock {
        background-color: #DD6B20;
    }
    </style>
""", unsafe_allow_html=True)

# Load data to check status
try:
    df, df_scaled, is_mock, audio_features = load_and_preprocess_data()
except Exception as e:
    df, df_scaled, is_mock, audio_features = None, None, True, []

# Layout split
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h1 class='main-title'>Sistema de Recomendação Musical</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Sistemas de Apoio à Decisão (SAD) — DCC166 (2026.1)</p>", unsafe_allow_html=True)
    
    # Display Banner Image
    banner_path = os.path.join("assets", "music_rec_banner.png")
    if os.path.exists(banner_path):
        st.image(banner_path, use_container_width=True)
        
    st.markdown("""
    ### 🎵 Sobre o Projeto
    Este sistema é um protótipo de **Filtro de Recomendação baseado em Conteúdo** (Content-Based Filtering) para plataformas de streaming de música. 
    Ele utiliza as características físicas e matemáticas do sinal de áudio das faixas (como dançabilidade, energia, valência e acústica) para encontrar e sugerir novas músicas alinhadas às preferências de um usuário, representadas por **faixas-semente**.
    
    A recomendação musical age como um sistema inteligente de suporte à decisão, reduzindo a sobrecarga de opções do usuário em catálogos com dezenas de milhões de faixas.
    """)

with col2:
    # Sidebar status or panel
    st.markdown("### 📊 Status do Sistema")
    
    if is_mock:
        st.markdown("""
        <div class='card' style='border-left-color: #DD6B20;'>
            <h4>⚠️ Modo de Demonstração (Dados Simulados)</h4>
            <p>O arquivo <code>data/dataset.csv</code> não foi encontrado ou está inacessível. O sistema está rodando com <b>500 faixas simuladas</b>.</p>
            <span class='badge badge-mock'>DADOS SIMULADOS</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.warning("""
        **Como carregar o catálogo completo (114k músicas):**
        1. Baixe o dataset no Kaggle: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset)
        2. Renomeie o arquivo baixado para `dataset.csv`
        3. Salve-o na pasta `data/` do projeto.
        4. O sistema irá carregar os dados reais automaticamente ao recarregar a página!
        """)
    else:
        st.markdown(f"""
        <div class='card' style='border-left-color: #38A169;'>
            <h4>✅ Catálogo Completo Carregado</h4>
            <p>O arquivo <code>data/dataset.csv</code> foi carregado com sucesso.</p>
            <p><b>Total de Faixas Únicas:</b> {len(df):,}</p>
            <p><b>Total de Gêneros:</b> {df['track_genre'].nunique()}</p>
            <span class='badge badge-real'>DADOS REAIS</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div class='card-purple'>
        <h4>👥 Equipe de Desenvolvimento</h4>
        <ul>
            <li><b>Felipe Lazzarini Cunha</b></li>
            <li><b>Lucas Castro Carvalho</b></li>
            <li><b>Pedro Detoni Pereira</b></li>
        </ul>
        <p style='font-size: 0.9rem; color: #A0AEC0;'>DCC / Universidade Federal de Juiz de Fora (UFJF)</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Informational section about features
st.markdown("## 🔍 Como as Recomendações São Geradas?")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    ### 📐 Abordagem Técnica
    O sistema utiliza dois métodos complementares de recomendação baseada em conteúdo:
    
    1. **Similaridade do Cosseno (Cosseno Centróide):**
       - As faixas-semente selecionadas pelo usuário são combinadas para formar um **Vetor de Perfil do Usuário** (a média das características de áudio normalizadas).
       - Calculamos o cosseno do ângulo entre este vetor-perfil e o vetor de cada música do catálogo. Músicas com maior similaridade (próxima de 1.0) são recomendadas.
       
    2. **Agrupamento K-Means + Cosseno:**
       - O catálogo de músicas é segmentado em grupos homogêneos (clusters) com base nas características de áudio.
       - Identificamos a qual cluster a faixa-semente pertence e recomendamos músicas daquele mesmo grupo, refinando a busca por proximidade ao perfil do usuário.
    """)

with col_right:
    st.markdown("""
    ### 🎛️ Características de Áudio Utilizadas
    - **Dançabilidade (Danceability):** Adequação da faixa para dançar (ritmo, estabilidade da batida, etc.).
    - **Energia (Energy):** Medida perceptual de intensidade e atividade.
    - **Valência (Valence):** Positividade musical transmitida pela faixa (alegre vs. triste).
    - **Acústica (Acousticness):** Confiança de se a faixa é acústica.
    - **Instrumentalidade (Instrumentalness):** Probabilidade de a faixa não conter vocais.
    - **Tempo (BPM):** Andamento estimado da música.
    - **Volume (Loudness):** Altura geral do som em decibéis (dB).
    - **Fala (Speechiness):** Presença de palavras faladas.
    - **Vivacidade (Liveness):** Presença de plateia na gravação.
    """)

# Instructions on what to do next
st.info("👈 Utilize a barra lateral para explorar as páginas! Acesse **🔍 Explorar Dados** para ver estatísticas ou **🎵 Recomendador** para começar a receber indicações.")
