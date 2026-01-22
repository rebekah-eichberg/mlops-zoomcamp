from pathlib import Path
import pickle

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI

app = FastAPI()

MODEL_PATH = Path("models/model.xgb")
DV_PATH = Path("models/preprocessor.b")

dv = None
booster = None


def load_model():
    global dv, booster
    with open(DV_PATH, "rb") as f_in:
        dv = pickle.load(f_in)

    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: dict):
    # expected input example:
    # {"PULocationID": "130", "DOLocationID": "205", "trip_distance": 3.66}

    pu = str(request["PULocationID"])
    do = str(request["DOLocationID"])
    trip_distance = float(request["trip_distance"])

    row = {
        "PU_DO": f"{pu}_{do}",
        "trip_distance": trip_distance,
    }

    X = dv.transform([row])
    dmatrix = xgb.DMatrix(X)
    pred = float(booster.predict(dmatrix)[0])

    return {"duration_minutes": pred}

@app.get("/")
def root():
    return {"service": "duration_prediction", "status": "ok"}