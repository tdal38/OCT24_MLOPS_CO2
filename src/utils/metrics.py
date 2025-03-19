import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import json

# Calcul de la RMSE :
def compute_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

# Calcul du R2 : 
def compute_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

# Renvoi d'un dictionnaire comportant les deux métriques : 
def compute_metrics(y_true, y_pred):
    metrics = {
        "rmse": compute_rmse(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred)
    }
    return metrics

# Enregistrement des metrics dans un fichier JSON : 
def save_metrics(metrics, filepath):
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)
