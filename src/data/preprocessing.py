#!/usr/bin/env python
# coding: utf-8

# Importation des librairies :
import os
import sys
import pandas as pd

# Importation de la librairie permettant la sauvegarde des fichiers de log : 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_script import setup_logging

# Importation de la configuration des chemins : 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

# Initialisation du logger : 
logger = setup_logging()

# Démarrage des logs :
logger.info("✅ Script de preprocessing démarré avec succès (preprocessing.py).")

# Chargement du fichier .csv :
raw_file_path = os.path.join(config.RAW_DIR, "DF_Raw.csv")

if not os.path.exists(raw_file_path):
    logger.error(f"❌ Le fichier {raw_file_path} n'existe pas.")
else:
    try:
        df = pd.read_csv(raw_file_path)
        logger.info(f"⚙️ Fichier .csv chargé avec succès : ({len(df)} lignes).")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du fichier .csv : {e}.")

# Vérification de la présence des colonnes nécessaires
    required_columns = ['Year', 'Mk', 'Cn', 'M (kg)', 'Ewltp (g/km)', 'Ft', 'Ec (cm3)', 'Ep (KW)', 'Erwltp (g/km)', 'Fc']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"❌ Colonne(s) manquante(s) dans le fichier .csv : {missing_columns}")
    else:
        logger.info("✅ Toutes les colonnes nécessaires sont présentes.")

# Suppression des doublons potentiels à travers les années. 
# On ne prend pas en compte la colonne "Cn" car de nombreuses variations d'orthographe existent pour un même modèle. 
subset_cols = [col for col in df.columns if col not in ['Cn', 'Year']]
initial_count = len(df)
df = df.drop_duplicates(subset=subset_cols)
logger.info(f"🔍 Suppression des doublons - Lignes restantes : {len(df)} (initialement {initial_count}).")

# Vérification de la colonne "Ft" - Travail de catégorisation nécessaire :
# Passage en minuscules des catégories en doublon :

if 'Ft' in df.columns:
    df['Ft'] = df['Ft'].str.lower()

# Suppression des lignes contenant un "unknown" (majoritairement composées de NaN) : 

    df = df[df['Ft'] != 'unknown']
    logger.info("🎯 Transformation de 'Ft' terminée - Suppression des 'unknown'.")
else:
    logger.warning("⚠️ Colonne 'Ft' absente, transformation non effectuée.")

# Rassemblement des variables :
# NB : Le dictionnaire peut être complété en cas de valeurs différentes dans le dataset utilisé : 

dico_fuel = {'petrol': 'Essence',
            'hydrogen': 'Hydrogene',
            'e85': 'Essence',
            'lpg': 'GPL',
            'ng': 'GPL',
            'ng-biomethane': 'Bio-Carburant',
            'diesel': 'Diesel',
            'petrol/electric': 'Hybride',
            'diesel/electric': 'Hybride',
            'electric': 'Electrique'
}

df['Ft'] = df['Ft'].replace(dico_fuel)
logger.info("🔄 Remplacement des valeurs spécifiques de 'Ft' terminé.")

# Mise de côté des modèles électriques (qui n'émettent pas directement de CO2) :

df = df[df['Ft'] != 'Electrique']

# Passage en majuscules de la colonne "Mk" : 
if 'Mk' in df.columns:
    df['Mk'] = df['Mk'].str.upper()
    logger.info("📝 Les marques ont été converties en majuscules.")
else:
    logger.warning("⚠️ Colonne 'Mk' absente, transformation non effectuée.")

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
    'VW': 'VOLKSWAGEN',
    '?KODA': 'SKODA',
    'ŠKODA': 'SKODA',
    'PSA AUTOMOBILES SA': 'PEUGEOT',
    'FCA ITALY': 'FIAT',
    'ALFA  ROMEO': 'ALFA ROMEO',
    'LANDROVER': 'LAND ROVER'
}
df['Mk'] = df['Mk'].replace(dico_marque)
logger.info("📝 Correction des marques terminée.")

# Suppression des marques trop peu connues : 

brands_to_delete = ['TRIPOD', 'API CZ', 'MOTO STAR', 'REMOLQUES RAMIREZ', 'AIR-BRAKES', 
                    'SIN MARCA', 'WAVECAMPER', 'CASELANI', 'PANDA']
df = df[~df['Mk'].isin(brands_to_delete)]
print(df[df['Mk'].isin(brands_to_delete)])
logger.info("📝 Suppression des marques peu connues.")

# Suppression des occurences trop faibles : 

def filter_brands(df, col='Mk', threshold=5):
    brands = df[col].tolist()
    unique_brands = df[col].unique().tolist()
    filtered_brands = [brand for brand in unique_brands if brands.count(brand) >= threshold]
    return filtered_brands

filtered_brands = filter_brands(df, col='Mk', threshold=5)
df = df[df['Mk'].isin(filtered_brands)]
logger.info("📝 Suppression des occurences trop faibles.")

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
    logger.info(f'📌 Nombre de lignes dont la valeur de "{target_col}" dépasse le seuil de {seuil} : {nb_outliers}.')
    
    # Suppression des lignes présentant des outliers :
    df_clean_no_outliers = df_merged[df_merged[diff_col] <= seuil]
    logger.info(f"🔄 Nombre de lignes après suppression des outliers : {len(df_clean_no_outliers)}.")
    
    return df_clean_no_outliers

# Liste des colonnes à filtrer successivement :
columns_to_filter = ['Ewltp (g/km)', 'Fc', 'M (kg)', 'Ec (cm3)', 'Ep (KW)', 'Erwltp (g/km)']

# On part du DataFrame initial (que l'on copie pour ne pas altérer l'original) :
df_temp = df.copy()

# Boucle sur chaque colonne pour appliquer le filtrage successif des outliers :
for col in columns_to_filter:
    df_temp = detect_outliers(df_temp, col)

logger.info(f"✅ Après filtrage successif, le nombre de lignes restantes est de : {len(df_temp)}.")

# Suppression des valeurs aberrantes après traitement :
df_clean_no_outliers_final = df_temp

# Suppression des colonnes ajoutées pour la détection de valeurs aberrantes afin d'éviter tout risque de fuite de données :
df_clean_no_outliers_final = df_clean_no_outliers_final[['Mk', 'Cn', 'M (kg)', 'Ewltp (g/km)', 'Ft', 'Ec (cm3)', 
                                                         'Ep (KW)', 'Erwltp (g/km)', 'Year', 'Fc']]

# Mise de côté des modèles hybrides trop peu représentés : 
df_clean_no_outliers_final = df_clean_no_outliers_final[df_clean_no_outliers_final['Ft'] != 'Hybride']
df_clean_no_outliers_final = df_clean_no_outliers_final[df_clean_no_outliers_final['Ft'] != 'Bio-Carburant']
logger.info("🔄 Mise de côté des valeurs de 'Ft' trop peu représentées terminée.")

# Encodage des variables catégorielles :
# Encodage de "Ft" :
df_clean_no_outliers_final = pd.get_dummies(df_clean_no_outliers_final, columns=['Ft'], prefix='Ft', drop_first=False)
bool_cols = df_clean_no_outliers_final.select_dtypes(include=['bool']).columns
df_clean_no_outliers_final[bool_cols] = df_clean_no_outliers_final[bool_cols].astype(int)
logger.info("✅ Encodage de la variable 'Ft' terminée.")

# Encodage de "Mk" : 
df_clean_no_outliers_final = pd.get_dummies(df_clean_no_outliers_final, columns=['Mk'], prefix='Mk', drop_first=False)
bool_cols = df_clean_no_outliers_final.select_dtypes(include=['bool']).columns
df_clean_no_outliers_final[bool_cols] = df_clean_no_outliers_final[bool_cols].astype(int)
logger.info("✅ Encodage de la variable 'Mk' terminée.")

# Enregistrement du fichier de données prétraitées : 
# Définir le chemin vers le dossier existant :
processed_dir = config.PROCESSED_DIR

# Création du dossier s'il n'existe pas :
try:
    os.makedirs(processed_dir, exist_ok=True)
    logger.info("🗂️ Dossier de sauvegarde des données prétraitées vérifié ou créé avec succès.")
except Exception as e:
    logger.error(f'❌ Erreur lors de la création du dossier "processed" : {e}')

# Création de la variable contenant le nom du fichier .csv à exporter : 
output_filename = "DF_Processed.csv"

# Construction du chemin complet vers le fichier dans le dossier "processed" existant : 
output_filepath = os.path.join(processed_dir, output_filename)

# Exportation du DataFrame en .csv : 
if not df_clean_no_outliers_final.empty:
    try:
        df_clean_no_outliers_final.to_csv(output_filepath, index=False)
        logger.info(f"📁 Fichier .csv enregistré avec succès : {output_filepath}")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'enregistrement du fichier .csv : {e}")
else:
    logger.error("❌ Le DataFrame est vide. Aucune sauvegarde n'a été effectuée.")