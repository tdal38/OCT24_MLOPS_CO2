import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(log_file='app.log', max_bytes=5*1024*1024, backup_count=5):
    """
    Configure le système de logging pour l'application.

    :param log_file: Nom du fichier de log.
    :param max_bytes: Taille maximale du fichier avant rotation (par défaut 5 Mo).
    :param backup_count: Nombre de fichiers de log de sauvegarde à conserver.
    :return: logger configuré.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Création du gestionnaire de fichier
    log_file = os.path.join(os.getcwd(), 'logs', 'recup_raw_data.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    
    # Format du log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Ajout du gestionnaire au logger
    logger.addHandler(file_handler)
    
    return logger

# Exemple d'utilisation du logger
#if __name__ == "__main__":
    # Initialiser le logger
    #logger = setup_logging(log_file='logs/app.log')

    # Exemples de messages de log
    #logger.debug("🔧 Message de debug - utile pour le développement.")
    #logger.info("ℹ️  Message d'information - tout fonctionne bien.")
    #logger.warning("⚠️  Avertissement - quelque chose pourrait mal tourner.")
    #logger.error("❌ Erreur - quelque chose n'a pas fonctionné.")
    #logger.critical("🔥 Problème critique - attention !")
