# Importation des librairies : 
import streamlit as st
import pandas as pd 
import numpy as np

# Pour l'entraînement des modèles
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Pour l'utilisation de DagsHub
import dagshub

# Pour l'utilisation de MLFlow
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Librairie pour utiliser les dossiers du système
import os
import shutil
import sys
from pathlib import Path

import config

# Librairie pour le téléchargement depuis Google Drive
import gdown

# Librairie pour mesurer le temps écoulé
import time
import asyncio

# Pour la requête SQL
import urllib.parse

# Pour le lancement de commande bash
import subprocess

# Pour le lancement des requêtes vers l'API
import requests

# Pour la gestion de l'authentification de l'API
import zipfile
from dotenv import load_dotenv

# Configuration de MLflow
mlflow.set_experiment("MLflow Streamlit")

# Fonction permettant de lancer la Pipeline DVC
def run_dvc_repro():
    try:
        result = subprocess.run(['dvc', 'repro', '--force'], check=True, capture_output=True, text=True)
        st.success('✅ Commande exécutée avec succès !')
        st.text(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error("❌ Une erreur est survenue lors de l'exécution de la commande DVC.")
        st.text(e.stderr)

# Fonction pour charger et prétraiter le dataset
#@st.cache_data
def load_and_preprocess_data():
    # Chargement du dataset :
    processed_file_path = os.path.join(config.PROCESSED_DIR, "DF_Processed.csv")
    df = pd.read_csv(processed_file_path)
    
    # Sélection des features (X) et de la target (y) : 
    baseline_features = ['M (kg)', 'Ec (cm3)', 'Ep (KW)', 'Erwltp (g/km)', 'Fc', 'Ft_Diesel', 'Ft_Essence']
    target = 'Ewltp (g/km)'

    X_baseline = df[baseline_features]
    y_baseline = df[target]
    
    # Split en train/test : 
    X_train, X_test, y_train, y_test = train_test_split(X_baseline, y_baseline, test_size=0.2, random_state=42)
    
    return df, X_train, X_test, y_train, y_test

# Fonction permettant un affichage dynamique de la taille du fichier
def format_size(bytes_size):
    """Formatage dynamique des tailles en KB, MB ou GB."""
    if bytes_size < 1024:
        return f"{bytes_size:.2f} B"
    elif bytes_size < 1024 ** 2:
        return f"{bytes_size / 1024:.2f} KB"
    elif bytes_size < 1024 ** 3:
        return f"{bytes_size / (1024 ** 2):.2f} MB"
    else:
        return f"{bytes_size / (1024 ** 3):.2f} GB"

def move_csv_files(src_folder, dest_folder):
    """
    Déplace tous les fichiers CSV du dossier racine vers le dossier de destination.

    :param src_folder: Chemin du dossier source.
    :param dest_folder: Chemin du dossier de destination.
    """
    # Vérifier l'existence du dossier source
    if not os.path.exists(src_folder):
        st.write(f"❌ Le dossier source '{src_folder}' n'existe pas.")
        return

    # Créer le dossier de destination s'il n'existe pas
    os.makedirs(dest_folder, exist_ok=True)

    # Parcourir uniquement le dossier racine (sans sous-dossiers)
    for file in os.listdir(src_folder):
        if file.endswith('.csv'):
            src_file = os.path.join(src_folder, file)
            dest_file = os.path.join(dest_folder, file)

            # Déplacer le fichier CSV
            shutil.move(src_file, dest_file)
            st.write(f"✅ Fichier déplacé : {src_file} -> {dest_file}")

    st.write(f"✅ Tous les fichiers CSV ont été déplacés vers : {dest_folder}")

# Fonction permettant la téléchargement des datasets avec une barre de progression
def download_file_with_progress(year):
    # Configuration de la requête
    base_url = "https://co2cars.apps.eea.europa.eu/tools/download"
    query_url = f"http://co2cars.apps.eea.europa.eu/10?source=%7B%22track_total_hits%22%3Atrue%2C%22query%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22constant_score%22%3A%7B%22filter%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22bool%22%3A%7B%22should%22%3A%5B%7B%22term%22%3A%7B%22year%22%3A{year}%7D%7D%5D%7D%7D%2C%7B%22bool%22%3A%7B%22should%22%3A%5B%7B%22term%22%3A%7B%22scStatus%22%3A%22Final%22%7D%7D%5D%7D%7D%5D%7D%7D%7D%7D%5D%7D%7D%2C%22display_type%22%3A%22tabular%22%7D"
    download_url = f"{base_url}?download_query={query_url}&download_format=csv"

    headers = {
        "Referer": "https://co2cars.apps.eea.europa.eu/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    # Stream the response to avoid loading everything in memory
    with requests.get(download_url, headers=headers, stream=True) as response:
        response.raise_for_status()
        
        # Taille totale du fichier
        total_size = int(response.headers.get('content-length', 0))
        total_size = int(total_size) if total_size is not None else 0

        block_size = 1024  # Taille de chaque chunk
        
        # Affichage de la barre de progression
        # progress_bar = st.progress(0)
        status_text = st.empty()
        downloaded_size = 0

        # Nom du fichier à sauvegarder        
        raw_dir = config.RAW_DIR
        os.makedirs(raw_dir, exist_ok=True)
        file_name = os.path.join(raw_dir, f"DF_Full_Raw_{year}.csv")

        with open(file_name, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
                downloaded_size += len(data)

                # Formatage de la taille téléchargée
                formatted_downloaded = format_size(downloaded_size)

                # Mise à jour de la progression
                if total_size > 0:
                    progress = int(downloaded_size * 100 / total_size)
                    # progress_bar.progress(min(progress, 100))
                    formatted_total = format_size(total_size)
                    status_text.text(f"🔄 Téléchargé : {formatted_downloaded} / {formatted_total} ({progress}%)")
                    status_text.text(f"🔄 Téléchargement : {progress}%")
                else:
                    # Si la taille totale est inconnue, afficher le nombre d'octets téléchargés
                    status_text.text(f"🔄 Téléchargement : {formatted_downloaded} téléchargés")

        return file_name

# Fonction permettant la téléchargement des datasets avec une requête SQL
def download_data_sql(selected_year):
    table_list = {
        '2021': 'co2cars_2021Fv24',
        '2022': 'co2cars_2022Fv26',
        '2023': 'co2cars_2023Fv28',
        '2024': 'co2cars_2024Fv30',
        '2025': 'co2cars_2025Fv32',
        '2026': 'co2cars_2026Fv34',
        '2027': 'co2cars_2027Fv36',
        '2028': 'co2cars_2028Fv38',
        '2029': 'co2cars_2029Fv40'
    }

    table = table_list[selected_year]

    query = f"""
    SELECT DISTINCT [Year] AS Year, Mk, Cn, [M (kg)], [Ewltp (g/km)], Ft, [Ec (cm3)], [Ep (KW)], [Erwltp (g/km)], Fc
    FROM [CO2Emission].[latest].[{table}]
    WHERE Mk IS NOT NULL 
      AND Cn IS NOT NULL 
      AND [M (kg)] IS NOT NULL
      AND [Ewltp (g/km)] IS NOT NULL
      AND Ft IS NOT NULL
      AND [Ec (cm3)] IS NOT NULL
      AND [Ep (KW)] IS NOT NULL
      AND [Erwltp (g/km)] IS NOT NULL
      AND [Year] IS NOT NULL
      AND Fc IS NOT NULL
    """

    encoded_query = urllib.parse.quote(query)
    page = 1
    records = []

    while True:
        url = f"https://discodata.eea.europa.eu/sql?query={encoded_query}&p={page}&nrOfHits=100000"
        response = requests.get(url)
        data = response.json()
        new_records = data.get("results", [])
        if not new_records:
            break
        records.extend(new_records)
        page += 1

    df = pd.DataFrame(records)
    return df

# Déplacement des fichiers puis suppression des dossiers
def move_contents_and_remove_folder(src_folder):
    """
    Déplace tous les fichiers et dossiers du dossier source vers le répertoire courant,
    puis supprime le dossier source.

    :param src_folder: Chemin du dossier source.
    """
    try:
        # Vérifier si le dossier source existe
        if not os.path.exists(src_folder):
            st.write(f"❌ Le dossier source {src_folder} n'existe pas.")
            return
        
        # Répertoire de destination (répertoire courant)
        dest_folder = os.getcwd()

        # Déplacement des fichiers et dossiers
        for item in os.listdir(src_folder):
            src_path = os.path.join(src_folder, item)
            dest_path = os.path.join(dest_folder, item)
            
            # Si le fichier ou dossier existe déjà, fusionner les dossiers ou écraser les fichiers
            if os.path.exists(dest_path):
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                    shutil.rmtree(src_path)
                else:
                    shutil.copy2(src_path, dest_path)
                    os.remove(src_path)
            else:
                shutil.move(src_path, dest_path)

        # Supprimer le dossier source s'il est vide
        if not os.listdir(src_folder):
            os.rmdir(src_folder)
            st.write(f"✅ Dossier source {src_folder} supprimé avec succès.")
        else:
            st.write(f"⚠️ Le dossier source {src_folder} n'a pas pu être supprimé car il n'est pas vide.")

        st.write(f"✅ Contenu déplacé vers {dest_folder}")

    except Exception as e:
        st.write(f"❌ Erreur : {e}")

# Charger le dataset
df, X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Définir les onglets
tabs = ["Téléchargement", "Exploration", "Entraînement", "Historiques", "Chargement", "Lancement", "Utilisateur Final"]

# Création des onglets
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tabs)

# Onglet 1 : Récupération des Datasets
with tab1:
    st.header("💽 Récupération du Projet via GitHub")

    if st.checkbox("Téléchargement du GitHub") :

        # Configuration du dossier GitHub (1 : dossier privé, 0 : dossier public)
        private = 0

        # Nom du fichier ZIP à sauvegarder
        zip_path = "c02_repo.zip"

        # Dossier d'extraction
        extract_to = "OCT24-MLE-CO2"
        
        # Téléchargement du dossier GitHub via GitHub si public
        if private == 0:
            # Mesurer le temps de début
            start_time = time.time()

            # URL du dépôt GitHub
            url_repo = "https://github.com/tdal38/OCT24-MLE-CO2/archive/refs/heads/main.zip"

            # Télécharger le fichier ZIP
            response = requests.get(url_repo, stream=True)
            if response.status_code == 200:
                with open(zip_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        file.write(chunk)
                # Mesurer le temps de fin
                end_time = time.time()
                
                # Calculer et afficher la durée
                execution_time = end_time - start_time
                
                st.write(f"✅ Téléchargement du dossier GitHub terminé en {execution_time} secondes !")
            else:
                st.write("❌ Erreur lors du téléchargement du dossier GitHub.")

        # Téléchargement du dossier GitHub via Google Drive si GitHub privé
        elif private == 1:
            # Mesurer le temps de début
            start_time = time.time()

            # ID du fichier Google Drive
            file_id_repo = "1Ne1VJWQX4ixc29jIscJApEBp-YzY0kcB"
            # URL de téléchargement direct
            url_repo = f"https://drive.google.com/uc?id={file_id_repo}"

            # Téléchargement du fichier
            gdown.download(url_repo, zip_path, quiet=False)

            # Mesurer le temps de fin
            end_time = time.time()

            # Calculer et afficher la durée
            execution_time = end_time - start_time

            # Confirmation de téléchargement - 5s
            st.write(f"✅ Téléchargement de {zip_path} s'est terminé en {round(execution_time,2)} secondes")

        # Mesurer le temps de début
        start_time = time.time()

        # Extraire le fichier ZIP
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)

            # Supprimer le fichier ZIP après extraction
            os.remove(zip_path)
            # Mesurer le temps de fin
            end_time = time.time()
            # Calculer et afficher la durée
            execution_time = end_time - start_time

            st.write(f"✅ Extraction terminée dans le dossier '{extract_to}' réalisé en {execution_time:.2f} seconde !")
        else:
            st.write("❌ Le fichier ZIP n'a pas été trouvé.")

        # Définir le chemin du dossier source
        source_folder = "./OCT24-MLE-CO2"

        # Vérifier si le dossier existe
        if not os.path.exists(source_folder):
            st.write("Le dossier source n'existe pas.")
        else:
            # Définir le dossier parent (le dossier précédent)
            parent_folder = os.path.dirname(os.path.abspath(source_folder))

            # Lister les fichiers et dossiers et les déplacer
            for item in os.listdir(source_folder):
                source_path = os.path.join(source_folder, item)
                destination_path = os.path.join(parent_folder, item)

                # Déplacer les fichiers et dossiers
                shutil.move(source_path, destination_path)

            # Supprimer le dossier vide
            os.rmdir(source_folder)

    if st.checkbox("Préparation des dossiers") :
        src = r"OCT24-MLE-CO2-main\NOV24-BDS-CO2"
        move_contents_and_remove_folder(src)
        st.write(f"✅ Traitement des dossiers et fichiers de GitHub terminée !")

        shutil.rmtree("OCT24-MLE-CO2-main")
        st.write(f"✅ Suppression du dossier inutile")

    st.header("📥 Récupération des Datasets")

# --- Téléchargement du CSV complet avec choix des dates --- # 
    st.write("Téléchargement du CSV complet avec choix des dates")
    # Sélection de l'année
    selected_year = st.selectbox("Sélectionnez l'année :", list(range(2010, 2030)))

    if st.button("Télécharger le fichier CSV"):
        # Mesurer le temps de début
        start_time = time.time()
        # Téléchargement du fichier
        file_path = download_file_with_progress(selected_year)

        if file_path:
            with open(file_path, "rb") as f:
                st.download_button(label="Télécharger le fichier", data=f, file_name=os.path.basename(file_path), mime="text/csv")
            st.success(f"✅ Fichier téléchargé : {file_path}")
        else:
            st.error("❌ Échec du téléchargement du fichier.")

        # Mesurer le temps de fin
        end_time = time.time()

        # Calculer et afficher la durée
        execution_time = end_time - start_time

        # Confirmation de téléchargement
        st.write(f"✅ Téléchargement de {file_path} s'est terminé en {round(execution_time,2)} secondes")

# --- Téléchargement du CSV avec requête SQL formatée --- # 
    st.write("Téléchargement du CSV avec requête SQL formatée")
    selected_year = st.selectbox('Choisissez une année :', [str(year) for year in range(2021, 2030)])

    if st.button('Télécharger les données'):
        with st.spinner('Téléchargement en cours...'):
            df = download_data_sql(selected_year)
            if not df.empty:
                raw_dir = config.RAW_DIR
                os.makedirs(raw_dir, exist_ok=True)
                output_filepath = os.path.join(raw_dir, f"DF_SQL_Raw_{selected_year}.csv")
                df.to_csv(output_filepath, index=False)
                st.success(f"✅ Les données ont été téléchargées avec succès et enregistrées dans {output_filepath}.")
            else:
                st.error("❌ Aucune donnée n'a été récupérée.")
    
    st.header("🗂️ Déplacement des fichiers CSV")
    if st.checkbox("Déplacer les fichiers"):
        source_directory = r"./"
        destination_directory = r"./data"

        move_csv_files(source_directory, destination_directory)

# Onglet 2 : Exploration du dataset
with tab2:
    st.header("📊 Exploration du Dataset")

    # Affichage d'un aperçu du dataset
    st.subheader("Aperçu du Dataset 📋")
    st.dataframe(df.head(10))  # Afficher les 10 premières lignes

# Onglet 3 : Entraînement des modèles
with tab3:
    st.header("🏋️‍♂️ Entraînement des Modèles")

    # Affichage du modèle
    st.subheader("Choix du modèle 📋")

    # Sélection du modèle
    model_choice = st.selectbox(
        "Choisissez un modèle de Machine Learning",
        ["Régression Linéaire", "KNN", "Random Forest"]
    )

    # Afficher le choix sélectionné
    st.write(f"🔍 Modèle sélectionné : **{model_choice}**")

    # Entraînement et Logging MLflow
    if st.button("Entraîner le modèle"):
        
        with mlflow.start_run():
            
            # Modèle sélectionné
            if model_choice == "Régression Linéaire":
                model = LinearRegression()
            elif model_choice == "KNN":
                model = KNeighborsRegressor()
            elif model_choice == "Random Forest":
                model = RandomForestRegressor()
            
            # Entraînement
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calcul des métriques
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Ajouter un tag d'info
            mlflow.set_tag("Training Info", f"Modèle {model_choice} entraîné sur DF2023")

            # Enregistrer le modèle MLflow
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="test_model",
                signature=signature,
                input_example=X_train,
                registered_model_name="tracking-quickstart",
            )

            # Affichage des résultats
            st.write(f"### 📊 Résultats du Modèle : {model_choice}")
            st.write(f"**R² :** {r2:.4f}") #.4f sert à avoir 4 chiffres apres la virgule
            st.write(f"**RMSE :** {rmse:.4f}") 

            # Affichage des valeurs réelles vs prédictions
            result_df = pd.DataFrame({"Valeur Réelle": y_test, "Prédiction": y_pred})
            st.dataframe(result_df.head(10))

# Onglet 4 : Historique des Runs
with tab4:
    st.header("📜 Historique des Runs MLflow")

    # Se connecter à MLflow et récupérer les runs triés par date
    mlflow.set_experiment("MLflow Streamlit")
    runs = mlflow.search_runs(order_by=["start_time DESC"])  # Tri des runs du plus récent au plus ancien

    if runs.empty:
        st.info("Aucun modèle enregistré pour l'instant. Lance un entraînement !")
    else:
        # Extraire les informations clefs des runs
        df_runs = runs[["run_id", "metrics.R2", "metrics.RMSE", "start_time", "tags.mlflow.runName"]]
        df_runs.columns = ["Run ID", "R² Score", "RMSE", "Date", "Nom du modèle"]

        # Sélectionner la dernière run par défaut
        latest_run_id = df_runs.iloc[0]["Run ID"]  # Premier élément (le plus récent)
        selected_run = st.selectbox("Affichage de la dernière Run :", list(df_runs["Nom du modèle"]), index=0)   

        # Afficher les détails de la run sélectionnée
        run_details = df_runs[df_runs["Nom du modèle"] == selected_run]
        st.write(f"📌 **Nom du modèle** : {run_details['Nom du modèle'].values[0]}")
        st.write(f"📂 **Run ID** : {run_details['Run ID'].values[0]}")
        st.write(f"📅 **Date** : {run_details['Date'].values[0]}")
        st.write(f"📊 **R² Score** : {run_details['R² Score'].values[0]:.4f}")
        st.write(f"📉 **RMSE** : {run_details['RMSE'].values[0]:.4f}")

# Onglet 5 : Chargement des modèles
with tab5:
    st.header("🚀 Charger un Modèle MLflow et Prédire")

    mlflow.set_experiment("MLflow Streamlit")
    runs = mlflow.search_runs(order_by=["start_time DESC"])  # Trier du plus récent au plus ancien

    if runs.empty:
        st.info("Aucun modèle enregistré pour l'instant. Lance un entraînement !")
    else:
        # Extraire les informations clés des modèles enregistrés
        df_models = runs[["run_id", "tags.mlflow.runName"]]
        df_models.columns = ["Run ID", "Nom du Modèle"]

        # Sélectionner un modèle via son nom
        selected_model_name = st.selectbox("🎯 Choisissez un Modèle :", df_models["Nom du Modèle"], index=0)
        selected_run_id = df_models[df_models["Nom du Modèle"] == selected_model_name]["Run ID"].values[0]  # Récupération du Run ID correspondant

        # Charger le modèle MLflow sélectionné
        model_uri = f"runs:/{selected_run_id}/test_model"
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        st.success(f"✅ Modèle '{selected_model_name}' chargé avec succès !")

        # Formulaire pour entrer les caractéristiques du véhicule
        st.write("### 🏎️ Entrez les caractéristiques du véhicule")
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                m_kg = st.number_input("Masse du véhicule (kg)", min_value=500, max_value=3000, step=1)
                ec_cm3 = st.number_input("Cylindrée (cm³)", min_value=500, max_value=6000, step=1)
            with col2:
                ep_kw = st.number_input("Puissance (kW)", min_value=20, max_value=500, step=1)
                erwltp = st.number_input("Réduction d'émissions WLTP (g/km)", min_value=0.0, max_value=3.5, step=0.01)

            fuel_consumption = st.number_input("Consommation de carburant (L/100km)", min_value=2.0, max_value=15.0, step=0.1)
            ft = st.selectbox("Type de carburant", ["Diesel", "Essence"])
            fuel_types = {"Diesel": [1, 0], "Essence": [0, 1]}
            ft_encoded = fuel_types[ft]

            # Construction de l'input (features utilisées lors de l'entraînement)
            input_values = [m_kg, ec_cm3, ep_kw, erwltp, fuel_consumption] + ft_encoded
            input_data_df = pd.DataFrame([input_values], columns=['M (kg)', 'Ec (cm3)', 'Ep (KW)', 'Erwltp (g/km)', 'Fc', 'Ft_Diesel', 'Ft_Essence'])
            input_data_df = input_data_df.astype({
                "M (kg)": int,
                "Ec (cm3)": int,
                "Ep (KW)": int,
                "Erwltp (g/km)": float,
                "Fc": float,
                "Ft_Diesel": int,
                "Ft_Essence": int
                })

            submitted = st.form_submit_button("🔎 Prédire")

        # Faire la prédiction lorsque le formulaire est soumis
        if submitted:
            prediction = loaded_model.predict(input_data_df)
            st.success(f"📊 **Prédiction des émissions de CO₂ : {prediction[0]:.2f} g/km**")

# Onglet 6 : Lancement de la Pipeline depuis DVC
with tab6:
    st.header("⚙️ Lancement de la Pipeline depuis DVC")

    if st.button('Exécuter la Pipeline'):
        with st.spinner('Exécution de la commande en cours...'):
            run_dvc_repro()

# Onglet 7 : Utilisateur final
with tab7:
    # Configuration des requêtes
    load_dotenv()
    API_URL = "http://127.0.0.1:8000"
    TOKEN_URL = f"{API_URL}/token"
    PREDICT_URL = f"{API_URL}/predict"

    # Interface
    st.header("🚗 Prédiction d'émissions de CO2")

    # Authentification
    st.sidebar.header("🔐 Authentification")
    username = st.sidebar.text_input("Nom d'utilisateur")
    password = st.sidebar.text_input("Mot de passe", type="password")
    login_button = st.sidebar.button("Se connecter")

    # Gestion du token
    if "token" not in st.session_state:
        st.session_state.token = None

    if login_button:
        try:
            response = requests.post(
                TOKEN_URL,
                data={"username": username, "password": password},
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            if response.status_code == 200:
                st.session_state.token = response.json()["access_token"]
                st.sidebar.success("Connexion réussie !")
            else:
                st.sidebar.error("Identifiants incorrects")
        except Exception as e:
            st.sidebar.error(f"Erreur de connexion: {str(e)}")

    # Formulaire de prédiction
    if st.session_state.token:
        with st.form("prediction_form_api"):
            st.header("📊 Caractéristiques du véhicule")
            
            col1, col2 = st.columns(2)
            with col1:
                M_kg = st.number_input("Poids (kg)", min_value=500, max_value=5000, value=1500)
                Ec_cm3 = st.number_input("Cylindrée (cm3)", min_value=500, max_value=8000, value=2000)
                Ep_KW = st.number_input("Puissance (KW)", min_value=20, max_value=1000, value=100)
            
            with col2:
                Erwltp_g_km = st.number_input("Emission RWltp (g/km)", min_value=0.0, max_value=500.0, value=100.0)
                Fc = st.number_input("Consommation carburant", min_value=0.0, max_value=20.0, value=6.5)
                fuel_type = st.selectbox("Type de carburant", ['Essence', 'Diesel', 'GPL', 'Hydrogene'])

            # Charger le fichier DF_Processed.csv pour récupérer les colonnes manquantes
            data_path = "data/processed/DF_Processed.csv"
            if not os.path.exists(data_path):
                st.error(f"Le fichier de données {data_path} n'existe pas.")
                st.stop()

            # Charger les données prétraitées pour récupérer la structure des colonnes
            df = pd.read_csv(data_path)
            X = df.drop(['Ewltp (g/km)', 'Cn', 'Year'], axis=1)
            all_columns = X.columns.tolist()  # Liste complète des colonnes d'origine
            marques = [col.replace('Mk_', '') for col in all_columns if col.startswith('Mk_')]
            Mk = st.selectbox("Marque (Mk)", marques)

            submit_button = st.form_submit_button("Prédire les émissions")

        if submit_button:
            try:
                # Création du payload
                payload = {
                    "M_kg": M_kg,
                    "Ec_cm3": Ec_cm3,
                    "Ep_KW": Ep_KW,
                    "Erwltp_g_km": Erwltp_g_km,
                    "Fc": Fc,
                    "fuel_type": fuel_type,
                    "Mk": Mk
                }
                
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                response = requests.post(PREDICT_URL, json=payload, headers=headers)
                
                if response.status_code == 200:
                    prediction = response.json()["prediction"]
                    st.success(f"### 🔍 Résultat : {prediction:.2f} g/km")
                else:
                    st.error(f"Erreur API: {response.text}")
                    
            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {str(e)}")

    else:
        st.warning("Veuillez vous authentifier pour accéder aux prédictions")