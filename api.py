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

# Authentification
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username == os.getenv("API_USER") and form_data.password == os.getenv("API_PASSWORD"):
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token = jwt.encode(
            {"sub": form_data.username, "exp": expire},
            SECRET_KEY,
            algorithm=ALGORITHM
        )
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Identifiants incorrects"
    )

# Endpoint de prédiction
# @app.post("/predict")
# async def predict(data: PredictionRequest, token: str = Depends(oauth2_scheme)):
#     try:
#         # Vérification de la marque
#         if data.Mk not in marques:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Marque invalide. Choisissez parmi: {marques}"
#             )

#         # Préparation des données
#         user_input = pd.DataFrame({
#             'M (kg)': [data.M_kg],
#             'Ec (cm3)': [data.Ec_cm3],
#             'Ep (KW)': [data.Ep_KW],
#             'Erwltp (g/km)': [data.Erwltp_g_km],
#             'Fc': [data.Fc]
#         })

#         # Encodage du carburant
#         for fuel in ['Essence', 'Diesel', 'GPL', 'Hydrogene']:
#             user_input[f'Ft_{fuel}'] = 1 if fuel == data.fuel_type else 0

#         # Encodage de la marque
#         for brand in marques:
#             user_input[f'Mk_{brand}'] = 1 if brand == data.Mk else 0

#         # Ajout des colonnes manquantes
#         for col in all_columns:
#             if col not in user_input.columns:
#                 user_input[col] = 0

#         # Réorganisation et normalisation
#         user_input = user_input[all_columns]
#         user_input_scaled = scaler.transform(user_input)

#         # Prédiction
#         prediction = model.predict(user_input_scaled)
#         return {"prediction": float(prediction[0])}
    
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=str(e)
#         )

# @app.post("/predict")
# async def predict(data: PredictionRequest, token: str = Depends(oauth2_scheme)):
#     try:
#         # Vérification de la marque
#         if data.Mk not in marques:
#             raise HTTPException(status_code=400, detail=f"Marque invalide. Choisissez parmi: {marques}")

#         # Construction de la ligne de données
#         row = [
#             data.M_kg, data.Ec_cm3, data.Ep_KW, data.Erwltp_g_km, data.Fc,
#             int(data.fuel_type == 'Essence'),
#             int(data.fuel_type == 'Diesel'),
#             int(data.fuel_type == 'GPL'),
#             int(data.fuel_type == 'Hydrogene')
#         ] + [int(brand == data.Mk) for brand in marques]

#         # Création du DataFrame
#         user_input = pd.DataFrame([row], columns=all_columns)
#         user_input_scaled = scaler.transform(user_input)
        
#         return {"prediction": float(model.predict(user_input_scaled)[0])}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

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