import pandas as pd
import kagglehub
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Variabili globali
df = None
model = None
embeddings = None

def init_engine():
    """Carico i database da INTERNET (Kaggle) + il tuo file LOCALE."""
    global df, model, embeddings
    print("🤖 IL MODELLO: Inizio il download dei dati da Internet...")
    
    try:
        # --- 1. IMDB ---
        print("📥 IL MODELLO: Scarico IMDB Top 1000...")
        path_imdb = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
        csv_imdb = os.path.join(path_imdb, "imdb_top_1000.csv")
        df_imdb = pd.read_csv(csv_imdb)
        
        df_imdb = df_imdb[['Series_Title', 'Overview', 'Genre']].rename(
            columns={'Series_Title': 'titolo', 'Overview': 'trama', 'Genre': 'genere'}
        )

        # --- 2. NETFLIX ---
        print("📥 IL MODELLO: Scarico Netflix...")
        path_netflix = kagglehub.dataset_download("shivamb/netflix-shows")
        csv_netflix = os.path.join(path_netflix, "netflix_titles.csv")
        df_netflix = pd.read_csv(csv_netflix)

        df_netflix = df_netflix[['title', 'description', 'listed_in']].rename(
            columns={'title': 'titolo', 'description': 'trama', 'listed_in': 'genere'}
        )

        # --- 3. IL TUO DATASET ---
        if os.path.exists("datasetmio.csv"):
            print("👤 IL MODELLO: Carico il tuo file personale...")
            df_mio = pd.read_csv("datasetmio.csv", sep=None, engine='python')
            
            colonne = df_mio.columns.tolist()
            if len(colonne) >= 3:
                df_mio = df_mio.rename(columns={
                    colonne[0]: 'titolo',
                    colonne[1]: 'trama',
                    colonne[2]: 'genere'
                })
            
            df_mio = df_mio[['titolo', 'trama', 'genere']]
            
            print("🔗 Unisco tutto...")
            df = pd.concat([df_imdb, df_netflix, df_mio], ignore_index=True)
        else:
            print("🔗 Unisco solo IMDB + NETFLIX...")
            df = pd.concat([df_imdb, df_netflix], ignore_index=True)
        
        # Pulizia
        df = df.dropna(subset=['trama']).drop_duplicates(subset=['titolo'])
        print(f"✅ Database caricato! Totale film: {len(df)}")

        # AI
        print("🧠 Avvio la rete neurale...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df['trama'].tolist(), show_progress_bar=True)
        
        print("🚀 TUTTO PRONTO!")
        return True

    except Exception as e:
        print(f"❌ ERRORE CRITICO: {e}")
        return False

def get_recommendation(user_query):
    """
    Logica DETERMINISTICA PULITA:
    - Niente casualità.
    - Niente messaggi extra di incertezza.
    """
    if df is None or model is None:
        return {"error": "Sto ancora caricando, pazienta un attimo!"}

    # 1. Calcolo vettori
    query_vec = model.encode([user_query])
    scores = cosine_similarity(query_vec, embeddings)[0]
    
    
    id_film = scores.argmax()
    alto_score = scores[id_film]
    
    film = df.iloc[id_film]
    
    # 3. Controllo soglia (0.24)
    if alto_score < 0.24:
        return {"found": False}
    
    # 4. Restituisco il risultato pulito
    return {
        "found": True,
        "titolo": film['titolo'],
        "trama": film['trama'],
        "genere": film['genere'],
        "score": str(round(alto_score, 2))
    }