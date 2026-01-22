from airflow.decorators import dag, task
from datetime import datetime

@dag(
    dag_id="train_duration",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "module03"],
    params={"year": 2023, "month": 1},
)
def train_duration():

    @task
    def train(year: int, month: int):
        from duration_prediction.train import main
        return main(year=year, month=month)

    # pull from Airflow "params" (set in UI when triggering)
    train("{{ params.year }}", "{{ params.month }}")

dag = train_duration()