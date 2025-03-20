#!/usr/bin/env python
# coding: utf-8

# Importation des librairies spécifiques à l'enregistrement du fichier final : 
import os
import json
from datetime import datetime
import sys

# Importation des librairies classiques :
import pandas as pd

# Importation de la configuration des chemins : 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

# Chemin vers le fichier metadata.json depuis le dossier initial : 
metadata_path = os.path.join(config.METADATA_DIR, "metadata.json")

# Lecture du fichier metadata :
with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Récupération du chemin du fichier raw tel qu'enregistré dans metadata : 
raw_data_path = metadata.get("raw_data")

# Chargement du fichier CSV :
df = pd.read_csv(raw_data_path)

# Suppression des doublons potentiels à travers les années. 
# On ne prend pas en compte la colonne "Cn" car de nombreuses variations d'orthographe existent pour un même modèle. 

subset_cols = [col for col in df.columns if col not in ['Cn', 'Year']]
df = df.drop_duplicates(subset=subset_cols)

# Vérification de la colonne "Ft" - Travail de catégorisation nécessaire :
# Passage en minuscules des catégories en doublon

df['Ft'] = df['Ft'].str.lower()

# Suppression des lignes contenant un "unknown" (majoritairement composées de NaN) : 

df = df[df['Ft'] != 'unknown']

# Rassemblement des variables
# NB : Le dictionnaire peut être complété en cas de valeurs différentes dans le dataset utilisé

dico_fuel = {'petrol': 'Essence',
             'hydrogen' : 'Essence',
             'e85': 'Essence',
             'lpg': 'Essence',
             'ng': 'Essence',
             'ng-biomethane' : 'Essence',
             'diesel': 'Diesel',
             'petrol/electric': 'Hybride',
             'diesel/electric': 'Hybride',
             'electric' : 'Electrique'
}

df['Ft'] = df['Ft'].replace(dico_fuel)

# Mise de côté des modèles électriques (qui n'émettent pas directement de CO2)

df = df[df['Ft'] != 'Electrique']

# Passage en majuscules de la colonne "Mk" : 
df['Mk'] = df['Mk'].str.upper()

# Liste des marques les plus répandues en Europe : 
target_brands = ['CITROEN', 'FORD', 'FIAT', 'RENAULT', 'MERCEDES', 'BMW', 'VOLKSWAGEN', 'ALPINE', 
                 'INEOS', 'LAMBORGHINI', 'TOYOTA', 'JAGUAR', 'GREAT WALL MOTOR', 'CATERHAM', 'PEUGEOT', 
                 'MAN', 'OPEL', 'ALLIED VEHICLES', 'IVECO', 'MITSUBISHI', 'DS', 'MAZDA', 'SUZUKI', 
                 'SUBARU', 'HYUNDAI', "AUDI", "NISSAN", "SKODA", "SEAT", "DACIA", "VOLVO", "KIA",
                 "LAND ROVER", "MINI", "PORSCHE", "ALFA ROMEO", "SMART", "LANCIA", "JEEP"
                 ]

# Fonction pour extraire les marques connues des chaînes de caractères : 
def extract_brand(value):
    for brand in target_brands:
        if brand in value:
            return brand
    return value
df['Mk'] = df['Mk'].apply(extract_brand)

# Correction des fautes de frappe : 
dico_marque = {
    'DS': 'CITROEN',
    'VW': 'VOLKSWAGEN',
    '?KODA': 'SKODA',
    'ŠKODA': 'SKODA',
    'PSA AUTOMOBILES SA': 'PEUGEOT',
    'FCA ITALY': 'FIAT',
    'ALFA  ROMEO': 'ALFA ROMEO',
    'LANDROVER': 'LAND ROVER'
}
df['Mk'] = df['Mk'].replace(dico_marque)

# Suppression des marques trop peu connues : 

brands_to_delete = ['TRIPOD', 'API CZ', 'MOTO STAR', 'REMOLQUES RAMIREZ', 'AIR-BRAKES', 
                    'SIN MARCA', 'WAVECAMPER', 'CASELANI', 'PANDA']
df = df[~df['Mk'].isin(brands_to_delete)]
print(df[df['Mk'].isin(brands_to_delete)])

# Suppression des occurences trop faibles : 

def filter_brands(df, col='Mk', threshold=5):
    brands = df[col].tolist()
    unique_brands = df[col].unique().tolist()
    filtered_brands = [brand for brand in unique_brands if brands.count(brand) >= threshold]
    return filtered_brands

filtered_brands = filter_brands(df, col='Mk', threshold=5)
df = df[df['Mk'].isin(filtered_brands)]

# Création d'une fonction pour détecter les valeurs aberrantes dans chaque colonne :

def detect_outliers(df, target_col, group_cols=["Cn", "Ft", "Year"]):
    # Calcul de la moyenne par groupe :
    stats = (
        df.groupby(group_cols)
          .agg(**{f'{target_col}_mean': (target_col, 'mean')})
          .reset_index()
    )
    
    # Fusion du DataFrame initial avec les statistiques calculées :
    df_merged = pd.merge(df, stats, on=group_cols, how="left")
    
    # Calcul de l'écart absolu entre la valeur et la moyenne :
    diff_col = f"diff_{target_col}"
    df_merged[diff_col] = (df_merged[target_col] - df_merged[f"{target_col}_mean"]).abs()
    
    # Calcul des quartiles et de l'IQR :
    q1 = df_merged[diff_col].quantile(0.25)
    q3 = df_merged[diff_col].quantile(0.75)
    iqr = q3 - q1
    
    # Calcul du seuil (Q3 + 1.5 * IQR) :
    seuil = (q3 + 1.5 * iqr).round(1)

    # Affichage du nombre d'outliers :
    nb_outliers = len(df_merged[df_merged[diff_col] >= seuil])
    print(f'Nombre de lignes dont la valeur de "{target_col}" dépasse le seuil de {seuil}: {nb_outliers}')
    
    # Suppression des lignes présentant des outliers :
    df_clean_no_outliers = df_merged[df_merged[diff_col] <= seuil]
    print(f"Nombre de lignes après suppression des outliers : {len(df_clean_no_outliers)}")
    
    return df_clean_no_outliers

# Liste des colonnes à filtrer successivement :
columns_to_filter = ['Ewltp (g/km)', 'Fc', 'M (kg)', 'Ec (cm3)', 'Ep (KW)', 'Erwltp (g/km)']

# On part du DataFrame initial (copie pour ne pas altérer l'original) :
df_temp = df.copy()

# Boucle sur chaque colonne pour appliquer le filtrage successif des outliers :
for col in columns_to_filter:
    print(col)
    df_temp = detect_outliers(df_temp, col)

print("\nAprès filtrage successif, le nombre de lignes restantes est de :", len(df_temp))

# Suppression des valeurs aberrantes après traitement :
df_clean_no_outliers_final = df_temp

# Suppression des colonnes ajoutées pour la détection de valeurs aberrantes afin d'éviter tout risque de fuite de données :
df_clean_no_outliers_final = df_clean_no_outliers_final[['Mk', 'Cn', 'M (kg)', 'Ewltp (g/km)', 'Ft', 'Ec (cm3)', 
                                                         'Ep (KW)', 'Erwltp (g/km)', 'Year', 'Fc']]

# Mise de côté des modèles hybrides trop peu représentés : 
df_clean_no_outliers_final = df_clean_no_outliers_final[df_clean_no_outliers_final['Ft'] != 'Hybride']

# Encodage des variables catégorielles :

# Encodage de "Ft" :
df_clean_no_outliers_final = pd.get_dummies(df_clean_no_outliers_final, columns=['Ft'], prefix='Ft', drop_first=False)
bool_cols = df_clean_no_outliers_final.select_dtypes(include=['bool']).columns
df_clean_no_outliers_final[bool_cols] = df_clean_no_outliers_final[bool_cols].astype(int)

# Encodage de "Mk" : 

df_clean_no_outliers_final = pd.get_dummies(df_clean_no_outliers_final, columns=['Mk'], prefix='Mk', drop_first=False)
bool_cols = df_clean_no_outliers_final.select_dtypes(include=['bool']).columns
df_clean_no_outliers_final[bool_cols] = df_clean_no_outliers_final[bool_cols].astype(int)

# Enregistrement du fichier de données prétraitées : 
# Définir les chemins vers les dossiers existants :
processed_dir = config.PROCESSED_DIR
metadata_dir = config.METADATA_DIR

# Créer les dossiers s'ils n'existent pas :
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(metadata_dir, exist_ok=True)

# Génération d'un timestamp au format YYYYMMDD_HHMMSS :
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"DF_Processed_{timestamp}.csv"

# Construction du chemin complet vers le fichier dans le dossier raw existant : 
output_filepath = os.path.join(processed_dir, output_filename)

# Enregistrement du DataFrame dans le fichier avec le nom dynamique : 
df_clean_no_outliers_final.to_csv(output_filepath, index=False)

# Définition du chemin complet vers le fichier de métadonnées :
metadata_file = os.path.join(metadata_dir, "metadata.json")

# Chargement du contenu existant du fichier metadata, s'il existe, sinon initialisation d'un dictionnaire vide : 
if os.path.exists(metadata_file):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
else:
    metadata = {}

# Ajout ou mise à jour de la clé pour les données prétraitées : 
metadata["processed_data"] = output_filepath

# Réécriture du fichier metadata avec les deux informations (pour ne pas écraser celles des données brutes) : 
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=4)