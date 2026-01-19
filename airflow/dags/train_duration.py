
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

PROJECT_PATH = "/opt/project"

with DAG(
    dag_id="train_duration",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops"],
) as dag:
    train = BashOperator(
        task_id="train",
        bash_command=(
            f"cd {PROJECT_PATH} && "
            "python 03-workflow-orchestration/duration_prediction.py "
            "--year 2023 --month 1"
        ),
        env={
            "MLFLOW_TRACKING_URI": "http://host.docker.internal:5000"
        },
    )