from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="hello_world",
    start_date=datetime(2025, 1, 1),
    schedule=None,      # manual trigger
    catchup=False,
) as dag:

    hello = BashOperator(
        task_id="hello_task",
        bash_command="echo 'Hello, Airflow!'",
    )