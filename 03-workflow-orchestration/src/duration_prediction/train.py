import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import root_mean_squared_error
import mlflow



models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


PROJECT_PATH = os.getenv("PROJECT_PATH", "/opt/project")
TAXI_DATA_FOLDER = os.getenv("TAXI_DATA_FOLDER", f"{PROJECT_PATH}/data")

def read_dataframe(year, month):
    taxi_folder = Path(TAXI_DATA_FOLDER)
    path =  taxi_folder / f"green_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(path)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    

    return df

def create_X(df, dv=None):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')    

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
       X = dv.transform(dicts)

    return X, dv


def train_model(X_train, y_train, X_val, y_val, dv):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_registry_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("nyc-taxi-experiment-serve-artifacts")
    
    with mlflow.start_run() as run:
            
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 10,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=10,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
        
        return run.info.run_id


def main(year, month):
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    
    print(f"TRAIN: {year}-{month:02d}  VAL: {next_year}-{next_month:02d}")
    df_train = read_dataframe(year=year, month=month)
    df_val = read_dataframe(year=next_year, month = next_month)
    
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)
    
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    
    result = train_model(X_train, y_train, X_val, y_val, dv)
    return result
    

if __name__ == "__main__":
    # use argparse to get year and month from command line
    import argparse
    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()
    
    output = main(year=args.year,month=args.month)
    print(output)
    # save run id to file
    
    with open("run_result.txt", "w") as f:
        f.write(str(output))

    