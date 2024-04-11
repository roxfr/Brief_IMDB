from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib


app = Flask(__name__)


##########################
# CHARGEMENT DES DONNEES #
##########################

data = pd.read_csv('data/5000_movies_cleaned.csv')

model = None

#############
# FONCTIONS #
#############

def predict_linear_regression(title):
    # Chargement du modèle de régression linéaire à partir du fichier
    model_file = 'data/model_linear_regression.pkl'
    model = joblib.load(model_file)

    # Extraction des données du film à partir du DataFrame en fonction du titre
    film_data = data[data['movie_title'] == title]

    # Vérification si les données du film sont vides
    if film_data.empty:
        # Si les données sont vides, renvoi une erreur 404 avec un message JSON
        return jsonify({'error': 'Film non trouvé'}), 404

    # Extraction des caractéristiques du film en supprimant le score IMDb et le titre du film
    features = film_data.drop(['imdb_score', 'movie_title'], axis=1).values

    # Standardisation des caractéristiques du film
    scaler = StandardScaler()  # Initialisation du StandardScaler
    features_scaled = scaler.fit_transform(features)  # Standardisation des caractéristiques

    # Prédiction du score IMDb du film à partir des caractéristiques standardisées
    score_predict = model.predict(features_scaled)

    # Extraction du score IMDb réel du film
    score_source = film_data['imdb_score'].values[0]

    # Renvoi du score IMDb prédit et du score IMDb réel
    return score_predict[0], score_source


def predict_random_forest(title):
    model_file = 'data/model_random_forest.pkl'
    model = joblib.load(model_file)

    film_data = data[data['movie_title'] == title]
    if film_data.empty:
        return jsonify({'error': 'Film non trouvé'}), 404

    features = film_data.drop(['imdb_score', 'movie_title'], axis=1).values
    score_predict = model.predict(features)
    score_source = film_data['imdb_score'].values[0]

    return score_predict[0], score_source

############
# ROUTAGES #
############

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET'])
def predict():
    # Récupération du titre et du type de modèle à partir des paramètres de la requête GET
    title = request.args.get('title')  # Titre du film à prédire
    model_type = request.args.get('model')  # Type de modèle à utiliser

    # Vérification du type de modèle et appel de la fonction de prédiction appropriée
    if model_type == 'linear_regression':
        score_predict, score_source = predict_linear_regression(title)  # régression linéaire
    elif model_type == 'random_forest':
        score_predict, score_source = predict_random_forest(title)  # random forest
    else:
        # Si le type de modèle n'est pas valide, renvoi une erreur 400
        return jsonify({'error': 'Modèle non valide'}), 400

    # Vérification du type de contenu de la requête et réponse appropriée
    if request.headers.get('Content-Type') == 'application/json':
        # Si la demande accepte JSON, renvoi les scores prédits et originaux au format JSON
        return jsonify({'score_predict': score_predict, 'score_source': score_source})
    else:
        # Sinon, renvoi une page HTML affichant les scores prédits et originaux
        return render_template('result.html', score_predict=score_predict, score_source=score_source)


if __name__ == '__main__':
    app.run(debug=True)
    