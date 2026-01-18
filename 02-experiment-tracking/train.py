import os
import pickle
import click
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from mlflow.tracking import MLflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("homework-experiment")

MLFLOW_TRACKING_URI = 'sqlite:///mlflow.db'

client = MLflowClient(tracking_uri=MLFLOW_TRACKING_URI)

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    
    mlflow.autolog()
    
    with mlflow.start_run():

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        
        # log the model
        mlflow.sklearn.log_model(
            sk_model = rf,
            name = "sklearn-model",
            input_example = X_train,
            registered_model_name = "sk-learn-random-forest-reg-model"
        )


if __name__ == '__main__':
    run_train()
