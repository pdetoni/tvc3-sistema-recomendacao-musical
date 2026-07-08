import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st

DATA_PATH = os.path.join("data", "dataset.csv")

# Audio features used for similarity calculation
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

def generate_mock_data(n_samples=500):
    """Generates synthetic Spotify-like data for testing and demonstration."""
    np.random.seed(42)
    genres = ["pop", "rock", "jazz", "classical", "hip-hop", "electronic", "reggae", "metal", "indie", "latin"]
    artists_list = ["Coldplay", "Taylor Swift", "The Weeknd", "Daft Punk", "Billie Eilish", "Miles Davis", "Ludwig van Beethoven", "Eminem", "Bob Marley", "Metallica"]
    album_list = ["Greatest Hits", "Discovery", "Future Nostalgia", "Midnights", "After Hours", "Kind of Blue", "Symphony No. 9", "The Eminem Show", "Legend", "Master of Puppets"]
    
    mock_data = []
    for i in range(n_samples):
        genre = np.random.choice(genres)
        artist = np.random.choice(artists_list)
        album = np.random.choice(album_list)
        track_name = f"Track {i+1} - {genre.capitalize()}"
        
        # Audio features based on genre profiles to make recommendations meaningful
        if genre == "classical":
            acousticness = np.random.uniform(0.8, 1.0)
            energy = np.random.uniform(0.0, 0.3)
            danceability = np.random.uniform(0.0, 0.3)
            instrumentalness = np.random.uniform(0.7, 1.0)
            tempo = np.random.uniform(60, 100)
            valence = np.random.uniform(0.1, 0.5)
        elif genre == "electronic" or genre == "pop":
            acousticness = np.random.uniform(0.0, 0.3)
            energy = np.random.uniform(0.6, 0.9)
            danceability = np.random.uniform(0.6, 0.9)
            instrumentalness = np.random.uniform(0.0, 0.5)
            tempo = np.random.uniform(110, 140)
            valence = np.random.uniform(0.5, 0.9)
        elif genre == "rock" or genre == "metal":
            acousticness = np.random.uniform(0.0, 0.2)
            energy = np.random.uniform(0.7, 1.0)
            danceability = np.random.uniform(0.3, 0.6)
            instrumentalness = np.random.uniform(0.0, 0.4)
            tempo = np.random.uniform(100, 160)
            valence = np.random.uniform(0.2, 0.7)
        else: # general defaults
            acousticness = np.random.uniform(0.1, 0.7)
            energy = np.random.uniform(0.3, 0.8)
            danceability = np.random.uniform(0.4, 0.8)
            instrumentalness = np.random.uniform(0.0, 0.3)
            tempo = np.random.uniform(80, 130)
            valence = np.random.uniform(0.3, 0.8)
            
        mock_data.append({
            "track_id": f"mock_id_{i:06d}",
            "artists": artist,
            "album_name": album,
            "track_name": track_name,
            "popularity": int(np.random.randint(10, 95)),
            "duration_ms": int(np.random.randint(120000, 300000)),
            "explicit": bool(np.random.choice([True, False], p=[0.1, 0.9])),
            "danceability": danceability,
            "energy": energy,
            "key": int(np.random.randint(0, 12)),
            "loudness": float(np.random.uniform(-20.0, -3.0)),
            "mode": int(np.random.choice([0, 1])),
            "speechiness": float(np.random.uniform(0.02, 0.2)),
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "liveness": float(np.random.uniform(0.05, 0.4)),
            "valence": valence,
            "tempo": tempo,
            "time_signature": int(np.random.choice([3, 4, 5])),
            "track_genre": genre
        })
        
    return pd.DataFrame(mock_data)

@st.cache_data
def load_and_preprocess_data():
    """
    Loads the Spotify Tracks Dataset from data/dataset.csv if it exists.
    Otherwise, generates mock data and returns a flag indicating it is mock data.
    Cleans data, removes duplicates, and fits standard scaler.
    """
    is_mock = False
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            # Remove unnamed index column if it exists (often Kaggle datasets have an index column)
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
        except Exception as e:
            st.error(f"Erro ao ler data/dataset.csv: {e}. Usando dados simulados.")
            df = generate_mock_data()
            is_mock = True
    else:
        df = generate_mock_data()
        is_mock = True
        
    # Data Cleaning: Drop rows where critical identifier columns are missing
    df = df.dropna(subset=["track_id", "track_name", "artists"])
    
    # Remove duplicate tracks by track_id
    df = df.drop_duplicates(subset=["track_id"]).reset_index(drop=True)
    
    # Clean artists column - if it contains ';;' separator (Kaggle Spotify dataset format)
    if not is_mock:
        df["artists"] = df["artists"].astype(str).str.replace(";;", ", ")
        
    # Fit the scaler on the audio features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[AUDIO_FEATURES])
    
    # Store the scaled features in a new dataframe
    df_scaled = pd.DataFrame(scaled_features, columns=AUDIO_FEATURES)
    # Re-attach IDs and key metadata for tracking
    df_scaled["track_id"] = df["track_id"]
    df_scaled["track_genre"] = df["track_genre"]
    
    return df, df_scaled, is_mock, AUDIO_FEATURES
