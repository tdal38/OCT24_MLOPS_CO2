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
    # Vérifier et créer le dossier des logs si nécessaire
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configuration du logger
    logger = logging.getLogger("AppLogger")
    logger.setLevel(logging.DEBUG)

    # Format du log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Handler pour l'enregistrement dans un fichier avec rotation
    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(formatter)

    # Handler pour l'affichage dans la console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Ajouter les handlers au logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

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
