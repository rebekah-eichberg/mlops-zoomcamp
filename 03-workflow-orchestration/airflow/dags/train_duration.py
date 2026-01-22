# from airflow.decorators import dag, task
# from datetime import datetime

# @dag(
#     dag_id="train_duration",
#     start_date=datetime(2025, 1, 1),
#     schedule=None,
#     catchup=False,
#     tags=["mlops", "module03"],
#     params={"year": 2023, "month": 1},
# )
# def train_duration():

#     @task
#     def train(year: int, month: int):
#         from duration_prediction.train import main
#         return main(year=year, month=month)

#     # pull from Airflow "params" (set in UI when triggering)
#     train("{{ params.year }}", "{{ params.month }}")

# dag = train_duration()

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from datetime import timedelta

@dag(
    dag_id="train_duration",
    schedule="@monthly",
    start_date=days_ago(90),
    catchup=True,
    tags=["mlops", "module03"],
)
def train_duration():

    @task
    def train(execution_date=None):
        from duration_prediction.train import main

        # Airflow gives execution_date as a pendulum datetime
        year = execution_date.subtract(months=2).year
        month = execution_date.subtract(months=2).month

        return main(year, month)

    train()

dag = train_duration()