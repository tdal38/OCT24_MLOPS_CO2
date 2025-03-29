from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import jwt
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import json

# Configuration
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Chargement du modèle et des données
model = joblib.load("models/RandomForest_Final.pkl")
df = pd.read_csv("data/processed/DF_Processed.csv")
X = df.drop(['Ewltp (g/km)', 'Cn', 'Year'], axis=1)
scaler = StandardScaler().fit(X)

# Récupération des features attendues
all_columns = X.columns.tolist()
marques = [col.replace('Mk_', '') for col in all_columns if col.startswith('Mk_')]

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Modèles Pydantic
class PredictionRequest(BaseModel):
    M_kg: int
    Ec_cm3: int
    Ep_KW: int
    Erwltp_g_km: float
    Fc: float
    fuel_type: str
    Mk: str

# Charger les utilisateurs depuis le fichier users.json
def load_users():
    with open("users.json", "r") as file:
        return json.load(file)["users"]

# Fonction pour vérifier si l'utilisateur existe et si le mot de passe est correct
def authenticate_user(username: str, password: str):
    users = load_users()
    for user in users:
        if user["username"] == username and user["password"] == password:
            return user  # Retourne l'utilisateur complet (incluant son rôle)
    return None

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if user:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token = jwt.encode(
            {"sub": user["username"], "role": user["role"], "exp": expire},
            SECRET_KEY,
            algorithm=ALGORITHM
        )
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Identifiants incorrects"
    )

@app.post("/predict")
async def predict(data: PredictionRequest, token: str = Depends(oauth2_scheme)):
    try:
        # Vérification de la marque
        if data.Mk not in marques:
            raise HTTPException(status_code=400, detail=f"Marque invalide. Choisissez parmi: {marques}")

        # Construction de la ligne de données
        row = [
            data.M_kg, data.Ec_cm3, data.Ep_KW, data.Erwltp_g_km, data.Fc,
            int(data.fuel_type == 'Essence'),
            int(data.fuel_type == 'Diesel'),
            int(data.fuel_type == 'GPL')
        ] + [int(brand == data.Mk) for brand in marques]

        # Vérification du nombre de colonnes
        if len(row) != len(all_columns):
            raise HTTPException(
                status_code=400,
                detail=f"Nombre de colonnes incorrect. Attendu: {len(all_columns)}, Reçu: {len(row)}, Nombre total de colonnes: {len(all_columns)}, Détail: 5 base + 4 fuel + {len(marques)} marques = {5 + 4 + len(marques)}"
            )

        # Création du DataFrame
        user_input = pd.DataFrame([row], columns=all_columns)
        user_input_scaled = scaler.transform(user_input)
        
        return {"prediction": float(model.predict(user_input_scaled)[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))