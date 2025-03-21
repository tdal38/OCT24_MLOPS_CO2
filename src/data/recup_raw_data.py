#!/usr/bin/env python
# coding: utf-8

# Importation des librairies pour la requête SQL et l'enregistrement du fichier final :
import requests
import urllib.parse

# Importation des librairies classiques :
import pandas as pd

# Importation des librairies spécifiques à l'enregistrement du fichier final : 
import os
import json
from datetime import datetime
import sys

# Importation de la configuration des chemins : 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

# Liste des tables disponibles sur le site de l'Agence Européenne à date : 
# NB : Il est possible de compléter cette liste avec les années postérieures 
# (années antérieures non compatibles avec le pre-processing actuel)
table_list = ['co2cars_2021Pv23', 'co2cars_2022Pv25', 'co2cars_2023Pv27']

# Définition de la requête et boucle for pour l'appliquer à tous les noms de table :
records = []

for table in table_list:
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
    response = requests.get(url)
    data = response.json()
    new_records = data.get("results", [])
    if not new_records:
      break
    records.extend(new_records)
    page += 1

# Transformation en DataFrame :
df = pd.DataFrame(records)

# Définir les chemins vers les dossiers existants :
raw_dir = config.RAW_DIR
metadata_dir = config.METADATA_DIR

# Créer les dossiers s'ils n'existent pas :
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(metadata_dir, exist_ok=True)

# Génération d'un timestamp au format YYYYMMDD_HHMMSS :
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"DF_Raw_{timestamp}.csv"

# Construction du chemin complet vers le fichier dans le dossier "raw" existant : 
output_filepath = os.path.join(raw_dir, output_filename)

# Enregistrement du DataFrame avec un nom dynamique : 
df.to_csv(output_filepath, index=False)

# Remplissage du fichier de métadonnées : 
metadata_file = os.path.join(metadata_dir, "metadata.json")
metadata = {"raw_data": output_filepath}
with open(metadata_file, "w") as f:
    json.dump(metadata, f)