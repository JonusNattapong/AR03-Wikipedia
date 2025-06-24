from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import requests

def trigger_nlp_pipeline(**kwargs):
    # เรียก API หรือรันสคริปต์ที่คุณต้องการ
    response = requests.post(
        "http://api:8000/generate/",
        json={"input_text": "Please generate summary of AI"}
    )
    print(response.json())

default_args = {
    'owner': 'you',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
    'start_date': datetime(2025, 1, 1),
}

with DAG(
    'nlp_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
) as dag:
    run_nlp = PythonOperator(
        task_id='run_nlp_pipeline',
        python_callable=trigger_nlp_pipeline,
        provide_context=True
    )
