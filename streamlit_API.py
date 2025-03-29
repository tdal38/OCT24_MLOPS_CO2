import streamlit as st
import requests
import jwt  # Pour décoder le token JWT
from dotenv import load_dotenv
import os
import pandas as pd
import subprocess

# Fonction permettant de lancer la Pipeline DVC
def run_dvc_repro():
    try:
        result = subprocess.run(['dvc', 'repro', '--force'], check=True, capture_output=True, text=True)
        st.success('✅ Commande exécutée avec succès !')
        st.text(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error("❌ Une erreur est survenue lors de l'exécution de la commande DVC.")
        st.text(e.stderr)

# Configuration des requêtes
load_dotenv()
API_URL = "http://127.0.0.1:8000"
TOKEN_URL = f"{API_URL}/token"
PREDICT_URL = f"{API_URL}/predict"

# Authentification
st.sidebar.header("🔐 Authentification")
username = st.sidebar.text_input("Nom d'utilisateur")
password = st.sidebar.text_input("Mot de passe", type="password")
login_button = st.sidebar.button("Se connecter")

if "token" not in st.session_state:
    st.session_state.token = None
    st.session_state.role = None

if login_button:
    try:
        response = requests.post(
            TOKEN_URL,
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        if response.status_code == 200:
            token = response.json()["access_token"]
            st.session_state.token = token
            decoded_token = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=["HS256"])
            st.session_state.role = decoded_token["role"]
            st.sidebar.success(f"Connexion réussie en tant que {decoded_token['role']} !")
        else:
            st.sidebar.error("Identifiants incorrects")
    except Exception as e:
        st.sidebar.error(f"Erreur de connexion: {str(e)}")

# Interface principale
if st.session_state.token:
    if st.session_state.role == "admin":
        st.header("⚙️ Lancement de la Pipeline depuis DVC")

        if st.button('Exécuter la Pipeline'):
            with st.spinner('Exécution de la commande en cours...'):
                run_dvc_repro()

    elif st.session_state.role == "user":
        # Interface
        st.header("🚗 Prédiction d'émissions de CO2")
        with st.form("prediction_form_api"):
            st.header("📊 Caractéristiques du véhicule")
            
            col1, col2 = st.columns(2)
            with col1:
                M_kg = st.number_input("Poids (kg)", min_value=500, max_value=5000, value=1500)
                Ec_cm3 = st.number_input("Cylindrée (cm3)", min_value=500, max_value=8000, value=2000)
                Ep_KW = st.number_input("Puissance (KW)", min_value=20, max_value=1000, value=100)
            
            with col2:
                Erwltp_g_km = st.number_input("Emission RWltp (g/km)", min_value=0.0, max_value=5.0, value=1.0)
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
        st.warning("Rôle inconnu. Accès refusé.")
else:
    st.warning("Veuillez vous authentifier pour accéder au contenu.")