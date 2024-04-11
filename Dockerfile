# Base Python
FROM python:3.11.7

# Répertoire de travail dans le conteneur
WORKDIR /app

# Fichiers nécessaires dans le conteneur
COPY requirements.txt .
COPY app.py .
COPY data data
COPY static static
COPY templates templates

# Dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Port sur lequel l'application Flask s'exécute
EXPOSE 5000

# Commande à exécuter
CMD ["python", "app.py"]
