import os

# Définir le dossier racine du projet à l'emplacement de config.py : 
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Définir les chemins relatifs à partir de PROJECT_ROOT : 
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
METADATA_DIR = os.path.join(PROJECT_ROOT, "metadata")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")