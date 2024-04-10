from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib


app = Flask(__name__)

# Chargement du modèle pré-entraîné
model = joblib.load('data/modele_random_forest.pkl')

# Chargement des données
data = pd.read_csv('data/5000_movies_cleaned.csv')

# Définition de la route pour la prédiction
# URL pour test => http://localhost:5000/predict?title=avatar
@app.route('/predict', methods=['GET'])
def predict_imdb_score():
    # Récupération du titre du film à partir des paramètres de la requête
    title = request.args.get('title')
    # Recherche des caractéristiques du film à partir de son titre
    film_data = data[data['movie_title'] == title]
    if film_data.empty:
        return jsonify({'error': 'Film non trouvé'}), 404
    # Extraction des caractéristiques du film
    features = film_data.drop(['imdb_score', 'movie_title'], axis=1).values
    # Prédiction
    score_predict = model.predict(features)
    # Récupération du score IMDB
    score_source = film_data['imdb_score'].values[0]
    # Retour
    return jsonify({'score_predict': score_predict[0], 'score_source': score_source})


if __name__ == '__main__':
    app.run(debug=True)
