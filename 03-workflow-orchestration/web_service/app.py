import pickle
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Ride(BaseModel):
    PULocationID: str
    DOLocationID: str
    trip_distance: float

with open("models/preprocessor.b", "rb") as f_in:
    dv = pickle.load(f_in)

booster = xgb.Booster()
booster.load_model("models/model.xgb")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(ride: Ride):
    pu_do = f"{ride.PULocationID}_{ride.DOLocationID}"
    X = dv.transform([{"PU_DO": pu_do, "trip_distance": ride.trip_distance}])
    pred = booster.predict(xgb.DMatrix(X))[0]
    return {"duration_minutes": float(pred)}