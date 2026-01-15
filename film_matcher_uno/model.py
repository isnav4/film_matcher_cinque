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
    """carico i due database da kaggle e anche il tuo."""
    global df, model, embeddings
    print("  il modello: inizio il downloaddd")
    
    try:
        
        print("il modello: scarico il dataset IMDB Top 1000...")
        path_imdb = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
        csv_imdb = os.path.join(path_imdb, "imdb_top_1000.csv")
        df_imdb = pd.read_csv(csv_imdb)
        
        df_imdb = df_imdb[['Series_Title', 'Overview', 'Genre']].rename(
            columns={'Series_Title': 'titolo', 'Overview': 'trama', 'Genre': 'genere'}
        )

        
        print("il modello: ora scarico il dataset di Netflix...")
        path_netflix = kagglehub.dataset_download("shivamb/netflix-shows")
        csv_netflix = os.path.join(path_netflix, "netflix_titles.csv")
        df_netflix = pd.read_csv(csv_netflix)

        df_netflix = df_netflix[['title', 'description', 'listed_in']].rename(
            columns={'title': 'titolo', 'description': 'trama', 'listed_in': 'genere'}
        )

        
        if os.path.exists("datasetmio.csv"):
            print("il modello: wow c'è anche un tuo dataset!! ora lo carico...")
            df_mio = pd.read_csv("datasetmio.csv", sep=None, engine='python')
            
            colonne = df_mio.columns.tolist()
            if len(colonne) >= 3:
                df_mio = df_mio.rename(columns={
                    colonne[0]: 'titolo',
                    colonne[1]: 'trama',
                    colonne[2]: 'genere'
                })
            
            df_mio = df_mio[['titolo', 'trama', 'genere']]
            
            print("adesso provo ad unire tutti i dataset")
            df = pd.concat([df_imdb, df_netflix, df_mio], ignore_index=True)
        else:
            print(" il tuo dataset non lo trovo! (oppure non mi piace) ora unisco solo IMDB e NETFLIX")
            df = pd.concat([df_imdb, df_netflix], ignore_index=True)
        
       
        df = df.dropna(subset=['trama']).drop_duplicates(subset=['titolo'])
        print(f" ho caricato il database ! ora conosco ben : {len(df)}film ")

        # AI
        print("ora avvio la mia incredibile(!!) rete neurale...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df['trama'].tolist(), show_progress_bar=True)
        
        print("ora sono davvero PRONTO!")
        return True

    except Exception as e:
        print(f"nooooo errore CRITICO: {e}")
        return False

def get_recommendation(user_query):
    """
    Logica DETERMINISTICA PULITA:
    - Niente casualità.
    - Niente messaggi extra di incertezza.
    """
    if df is None or model is None:
        return {"error": "sto ancora caricando, TU quanto ci metteresti??!"}

   
    query_vec = model.encode([user_query])
    scores = cosine_similarity(query_vec, embeddings)[0]
    
    
    id_film = scores.argmax()
    alto_score = scores[id_film]
    
    film = df.iloc[id_film]
    
    
    if alto_score < 0.24:
        return {"found": False}
    
    
    return {
        "found": True,
        "titolo": film['titolo'],
        "trama": film['trama'],
        "genere": film['genere'],
        "score": str(round(alto_score, 2))
    }