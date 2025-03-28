# Utiliser une image Python officielle
FROM python:3.12

# Définir le répertoire de travail dans le container
WORKDIR /app

# Copier uniquement les fichiers nécessaires
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY . /app/

# Exécuter Streamlit au lancement du container
CMD ["streamlit", "run", "stream3.py", "--server.port=8502", "--server.address=0.0.0.0"]
