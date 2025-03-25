#!/usr/bin/env python
# coding: utf-8

# Importation des librairies :
import requests
import urllib.parse
import pandas as pd
import os
import sys
import json

# Importation de la fonction permettant la sauvegarde des fichiers de log :
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_script import setup_logging


# Importation de la configuration des chemins : 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

# Initialisation du logger :
logger = setup_logging()

# Démarrage des logs :
logger.info("✅ Script de téléchargement des données démarré avec succès (recup_raw_data.py).")

# Liste des tables disponibles sur le site de l'Agence Européenne à date : 
# NB : Il est possible de compléter cette liste avec les années postérieures 
# (années antérieures non compatibles avec le pre-processing actuel)
table_list = ['co2cars_2021Pv23', 'co2cars_2022Pv25', 'co2cars_2023Pv27']

# Définition de la requête et boucle for pour l'appliquer à tous les noms de table :
records = []

for table in table_list:

  logger.info(f"📥 Début de téléchargement pour la table : {table}")

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

# Encodage de la requête pour l'inclure dans l'URL :
  encoded_query = urllib.parse.quote(query)

# Initialisation :
  page = 1

# Boucle while pour parcourir toutes les pages de l'API : 
# NB : Celle-ci s'arrête quand il n'y a plus de réponse.

  while True:
    url = f"https://discodata.eea.europa.eu/sql?query={encoded_query}&p={page}&nrOfHits=100000"
    logger.info(f"🌐 Requête envoyée.")
    try:
      response = requests.get(url)
      if response.status_code != 200:
          logger.error(f"❌ Erreur de requête : Status code {response.status_code}")
          break
      try:
          data = response.json()
      except json.JSONDecodeError as e:
          logger.error(f"❌ Erreur de décodage JSON : {e}")
          break
            
      new_records = data.get("results", [])
      if not new_records:
          logger.info(f"📌 Aucun enregistrement supplémentaire trouvé pour {table} (page {page}).")
          break

      records.extend(new_records)
      logger.info(f"✅ Page {page} téléchargée - {len(new_records)} enregistrements récupérés.")
            
      page += 1  # Passage à la page suivante

    except requests.RequestException as e:
        logger.error(f"❌ Erreur lors de la requête HTTP : {e}")
        break

# Transformation en DataFrame :
if records:
    df = pd.DataFrame(records)
    logger.info(f"✅ DataFrame créé avec succès - {len(df)} lignes téléchargées.")
else:
    logger.error("❌ Aucune donnée récupérée. Impossible de créer le DataFrame.")

# Enregistrement du fichier :
# Définition du chemins vers le dossier "raw" (pour exporter la base de données brute) :
raw_dir = config.RAW_DIR

# Création du dossier s'il n'existe pas :
try:
    os.makedirs(raw_dir, exist_ok=True)
    logger.info("🗂️ Dossier de sauvegarde des données brutes vérifié ou créé avec succès.")
except Exception as e:
    logger.error(f'❌ Erreur lors de la création du dossier "raw" : {e}')

# Création de la variable contenant le nom du fichier .csv à exporter : 
output_filename = "DF_Raw.csv"

# Construction du chemin complet vers le fichier dans le dossier "raw" existant : 
output_filepath = os.path.join(raw_dir, output_filename)

# Exportation du DataFrame en .csv : 
if not df.empty:
    try:
        df.to_csv(output_filepath, index=False)
        logger.info(f"📁 Fichier .csv enregistré avec succès : {output_filepath}")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'enregistrement du fichier .csv : {e}")
else:
    logger.error("❌ Le DataFrame est vide. Aucune sauvegarde n'a été effectuée.")