from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib


app = Flask(__name__)

# Chargement des données
data = pd.read_csv('data/5000_movies_cleaned.csv')

# Chargement du modèle pré-entraîné
model = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET'])
def predict():
    title = request.args.get('title')
    model_type = request.args.get('model')
    
    if model_type == 'linear_regression':
        model_file = 'data/modele_linear_regression.pkl'
    elif model_type == 'random_forest':
        model_file = 'data/modele_random_forest.pkl'
    else:
        return jsonify({'error': 'Modèle non valide'}), 400

    global model
    model = joblib.load(model_file)

    film_data = data[data['movie_title'] == title]
    if film_data.empty:
        return jsonify({'error': 'Film non trouvé'}), 404

    features = film_data.drop(['imdb_score', 'movie_title'], axis=1).values
    score_predict = model.predict(features)
    score_source = film_data['imdb_score'].values[0]

    if request.headers.get('Content-Type') == 'application/json':
        return jsonify({'score_predict': score_predict[0], 'score_source': score_source})
    else:
        return render_template('result.html', score_predict=score_predict[0], score_source=score_source)


if __name__ == '__main__':
    app.run(debug=True)
