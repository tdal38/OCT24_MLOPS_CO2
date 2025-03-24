#!/usr/bin/env python
# coding: utf-8

# Importation des librairies : 

import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

import os
import joblib
import sys

import dagshub
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Connexion avec MLFlow pour le suivi des expérimentations et l'enregistrement du modèle : 
dagshub.init(
    repo_owner="tiffany.dalmais",
    repo_name="OCT24_MLOPS_CO2",
    mlflow=True
)
mlflow.autolog()
with mlflow.start_run():

    # Importation de la configuration des chemins : 
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    import config

    # Chargement du nom du fichier de données prétraitées :
    processed_file_path = os.path.join(config.PROCESSED_DIR, "DF_Processed.csv")
    df = pd.read_csv(processed_file_path)

    # Séparation de X et y :
    X = df.drop(['Ewltp (g/km)', 'Cn', 'Year'], axis=1)
    y = df['Ewltp (g/km)']

    # Split train/test :
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisation :
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modèle final - RandomForestRegressor :
    model_final = RandomForestRegressor(bootstrap=False, max_features=0.75, min_samples_leaf=1,
                                        min_samples_split=9, n_estimators=100, random_state=42)
    model_final.fit(X_train_scaled, y_train)
    results_model_final = model_final.predict(X_test_scaled)

    # Enregistrement du modèle dans MLFlow : 
    params = {
    "bootstrap": False,
    "max_features": 0.75,
    "min_samples_leaf": 1,
    "min_samples_split": 9,
    "n_estimators": 100,
    "random_state": 42
    }
    signature = infer_signature(X_test_scaled, results_model_final)

    # Affichage et enregistrement des metrics : 
    # Import des fonctions créées dans le fichier metrics.py : 
    from src.utils.metrics import compute_metrics, save_metrics

    # Calcul et affichage des metrics : 
    metrics = compute_metrics(y_test, results_model_final)
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²  : {metrics['r2']:.4f}")

    # Enregistrement des metrics : 
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    metrics_file = os.path.join(config.OUTPUTS_DIR, "metrics.json")
    save_metrics(metrics, metrics_file)

    # Enregistrement des paramètres et metrics : 
    mlflow.log_params(params)
    mlflow.log_metric("rmse", metrics["rmse"])
    mlflow.log_metric("r2", metrics["r2"])
    
    # Enregistrement du modèle : 
    mlflow.sklearn.log_model(
        sk_model=model_final,
        artifact_path="sklearn-model",
        signature=signature,
        registered_model_name="RandomForest_Final",
    )

    # Analyse des erreurs : 
    df_results_final = pd.DataFrame({'y_true': y_test, 'y_pred': results_model_final})
    df_results_final['error'] = abs(df_results_final['y_true'] - df_results_final['y_pred'])
    seuil = 20
    outliers = df_results_final[df_results_final['error'] > seuil]
    print(outliers.describe())

    # Création de la variable contenant le nom du modèle : 
    model_filename = "RandomForest_Final.pkl"

    # Enregistrement du modèle : 
    # Définition du chemin :
    models_dir = config.MODELS_DIR

    # Création du dossier s'il n'existe pas :
    os.makedirs(models_dir, exist_ok=True)

    # Construction du chemin complet vers le fichier dans le dossier "models" existant :
    model_path = os.path.join(models_dir, model_filename)

    # Enregistrement du modèle entraîné : 
    joblib.dump(model_final, model_path)