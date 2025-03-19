#!/usr/bin/env python
# coding: utf-8

# Import des librairies : 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, BaggingRegressor

import json
import os
import joblib
from datetime import datetime
import sys

# Importation de la configuration des chemins : 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

# Chargement du nom du fichier à partir du fichier de métadonnées :
# Chemin vers le fichier metadata.json depuis le dossier initial : 
metadata_path = os.path.join(config.METADATA_DIR, "metadata.json")

# Lecture du fichier metadata :
with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Récupération du chemin du fichier raw tel qu'enregistré dans metadata : 
processed_data_path = metadata.get("processed_data")

# Chargement du fichier CSV :
df = pd.read_csv(processed_data_path)

# Séparation de X et y :
X = df.drop(['Ewltp (g/km)', 'Cn', 'Year'], axis=1)
y = df['Ewltp (g/km)']

# Split train/test :
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation :
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle final: RandomForestRegressor
model_final = RandomForestRegressor(bootstrap=False, max_features=0.75, min_samples_leaf=1,
                                    min_samples_split=9, n_estimators=100, random_state=42)
model_final.fit(X_train_scaled, y_train)
results_model_final = model_final.predict(X_test_scaled)

# Calcul RMSE
mse_mf = mean_squared_error(y_test, results_model_final)
rmse_mf = np.sqrt(mse_mf)
print("RMSE:", rmse_mf)

# Analyse erreurs
df_results_final = pd.DataFrame({'y_true': y_test, 'y_pred': results_model_final})
df_results_final['error'] = abs(df_results_final['y_true'] - df_results_final['y_pred'])
seuil = 20
outliers = df_results_final[df_results_final['error'] > seuil]
print(outliers.describe())

# Générer un nom de fichier dynamique pour le modèle :
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"RandomForest_{timestamp}.pkl"

# Construction du chemin complet vers le fichier dans le dossier models existant :
model_path = os.path.join(config.MODELS_DIR, model_filename)

# Enregistrement du modèle entraîné : 
joblib.dump(model_final, model_path)

# Chemin complet vers le fichier metadata.json : 
metadata_file = os.path.join(config.METADATA_DIR, "metadata.json")

# Charger le contenu existant de metadata.json s'il existe, sinon initialiser un dictionnaire vide : 
if os.path.exists(metadata_file):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
else:
    metadata = {}

# Mise à jour ou ajout de la clé pour le modèle entraîné : 
metadata["trained_model"] = model_path

# Réécriture du fichier metadata en conservant les autres informations : 
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=4)