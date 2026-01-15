from flask import Flask, render_template, request, jsonify
import model
import csv                 
import os                   
from datetime import datetime 

app = Flask(__name__)


model.init_engine()

file_feedback = 'dataset_feedback.csv'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    dati_json = request.get_json()
    
  
    frase_utente = dati_json.get('messaggio', '')
    
    if not frase_utente:
        return jsonify({"risposta": "please, try with a different word!"})
    
    risultato = model.get_recommendation(frase_utente)
    
    if risultato.get("error"):
        html_response = "<div class='msg bot error'>errooooore CRITICO del server.</div>"
    
    elif not risultato["found"]:
        html_response = "<div>I didn't found films with these words.</div>"
    
    else:
        
        html_response = f"""
        <div class='movie-card'>
            <h3>{risultato['titolo']}</h3>
            <span class='badge'>{risultato['genere']}</span>
            
            <p>{risultato['trama']}</p>
            
            <small>Score rilevanza: {risultato['score']}</small>
            
            <div class='feedback-buttons'>
                <span class='feedback-text'>Ti piace?</span>
                <button class='btn-vote up' onclick='vote("POS", "{risultato['titolo']}", "{frase_utente}", this)'>Si</button>
                <button class='btn-vote down' onclick='vote("NEG", "{risultato['titolo']}", "{frase_utente}", this)'>No</button>
            </div>
        </div>
        """
    
    return jsonify({"risposta": html_response})


@app.route('/feedback', methods=['POST'])
def feedback():
    informazioni = request.json
    
    esiste_gia = os.path.isfile(file_feedback)
    
    try:
        with open(file_feedback, mode='a', newline='', encoding='utf-8') as file:
            scrittore = csv.writer(file)
            
            if not esiste_gia:
                scrittore.writerow(['Data', 'Query_Utente', 'Film_Consigliato', 'Voto'])
            
           
            scrittore.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                informazioni['query'],
                informazioni['film'],
                informazioni['voto']
            ])
            
        print("DEBUG: ora il tuo feedback è stato salvato.")
        return jsonify({"status": "ok"})
        
    except Exception as e:
        print(f"errore: {e}")
        return jsonify({"status": "error"}), 500

if __name__ == '__main__':
    app.run(debug=True)