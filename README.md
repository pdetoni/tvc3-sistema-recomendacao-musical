# 🎵 Sistema de Recomendação Musical (TVC3)

Repositório contendo o protótipo de **Sistema de Recomendação Musical** desenvolvido para o Trabalho de Verificação de Conhecimento (TVC3) da disciplina **Sistemas de Apoio à Decisão (SAD)** do Departamento de Ciência da Computação (DCC) da Universidade Federal de Juiz de Fora (UFJF).

**Equipe:**
- Felipe Lazzarini Cunha
- Lucas Castro Carvalho
- Pedro Detoni Pereira

---

## 🎨 Apresentação do App

O projeto consiste em um aplicativo multipáginas desenvolvido em **Python + Streamlit**, que funciona como um **Sistema de Apoio à Decisão** para reduzir a sobrecarga de escolhas em catálogos musicais.

### 📐 Abordagem Técnica
O sistema implementa a recomendação baseada em conteúdo em duas abordagens distintas:
1. **Similaridade do Cosseno (Cosseno Centróide):** Consolida as faixas-semente selecionadas pelo usuário em um vetor de perfil médio e busca músicas com maior similaridade vetorial.
2. **Agrupamento via K-Means:** Segmenta o catálogo de músicas em clusters distintos e restringe a busca e similaridade a músicas do mesmo grupo das sementes, garantindo coesão de estilo.

### 📊 Estrutura do App Streamlit
- **Página Inicial:** Introdução teórica das características físicas de áudio extraídas (dançabilidade, energia, valência, etc.).
- **1. Explorar Dados:** Dashboard interativo de análise exploratória (EDA) com histogramas de features, heatmaps de correlação e gráficos de dispersão.
- **2. Recomendador:** Seleção de até 5 sementes com visualização radar em tempo real comparando a assinatura de áudio do usuário contra as recomendações.
- **3. Avaliação:** Mapeamento de métricas como *Intra-List Similarity* (ILS), *Catalog Coverage*, experimentos de *Hold-out (Hit Rate)* e índice de Silhueta do K-Means.

---

## 📂 Arquitetura do Repositório

```
tvc3/
├── .streamlit/
│   └── config.toml          ← Configuração do tema visual escuro
├── core/
│   ├── data_loader.py       ← Carga, limpeza e normalização (StandardScaler)
│   ├── recommender.py       ← Motores de similaridade do Cosseno e K-Means
│   └── evaluation.py        ← Cálculo de ILS, Coverage, Hit Rate e Silhueta
├── data/
│   └── README.md            ← Instruções para posicionamento da base
├── docs/
│   ├── relatorio.tex        ← Relatório acadêmico estruturado em LaTeX (SBC)
│   └── sbc-template.bib     ← Base de dados bibliográficos
├── imagens/
│   └── fluxo_sistema.png    ← Diagrama de fluxo de dados do SAD
└── pages/
    ├── 1_🔍_Explorar_Dados.py
    ├── 2_🎵_Recomendador.py
    └── 3_📊_Avaliacao.py
```

---

## 🚀 Como Executar Localmente

### 1. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 2. Configurar o Dataset
O sistema utiliza o **Spotify Tracks Dataset** disponível no Kaggle:
1. Baixe o dataset em: [maharshipandya/spotify-tracks-dataset](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset)
2. Extraia o zip e salve o arquivo como `dataset.csv` dentro da pasta `data/` do projeto (`tvc3/data/dataset.csv`).
*(Nota: Caso a base não esteja na pasta, o app iniciará automaticamente em **Modo de Demonstração** usando dados sintéticos gerados em tempo real).*

### 3. Rodar a aplicação
```bash
python -m streamlit run app.py
```
Acesse no navegador através de: **http://localhost:8501**
