from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from sqlalchemy import create_engine
from airflow import DAG
from airflow.decorators import task
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.providers.postgres.hooks.postgres import PostgresHook
from botocore.exceptions import ClientError

from calculate_batch_features import (
    build_batch_features,
    validate_features,
)

DAG_ID = "batch_features"
POSTGRES_CONN_ID = "student_db_connection"
S3_CONN_ID = "student_s3"
DEFAULT_OUTPUT_DIR = "/tmp/batch_features"


def get_postgres_uri() -> str:
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    return hook.get_uri()


def get_s3_client_and_bucket() -> tuple[Any, str]:
    connection = BaseHook.get_connection(S3_CONN_ID)
    extras = connection.extra_dejson

    bucket = extras.get("bucket")
    if not bucket:
        raise ValueError(f"Missing `bucket` in extras for connection `{S3_CONN_ID}`")

    
    aws_access_key_id = connection.login or extras.get("aws_access_key_id")
    aws_secret_access_key = connection.password or extras.get("aws_secret_access_key")

    client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=extras.get("endpoint_url", "https://storage.yandexcloud.net"),
    )
    return client, bucket


with DAG(
    dag_id=DAG_ID,
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["batch-features"],
) as dag:

    @task(task_id="build_and_upload_features")
    def build_and_upload_features() -> dict[str, str | int]:
    
        run_date = Variable.get("batch_features_run_date")  #Берём run_date из Airflow Variables

        pg_uri = get_postgres_uri()                         #Создаём engine из Airflow Connection
        engine = create_engine(pg_uri)

        run_date_ts = pd.Timestamp(run_date)                #Считаем батчи
        features = build_batch_features(run_date_ts, engine)

        validate_features(features, run_date_ts)                 #Валидация

        local_dir = Path(DEFAULT_OUTPUT_DIR) / f"run_date={run_date}"
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / "batch_features.csv"
        features.to_csv(local_path, index=False)

        s3_client, s3_bucket = get_s3_client_and_bucket()   #загружаем в s3
        s3_key = f"run_date={run_date}/batch_features.csv"
        s3_client.upload_file(str(local_path), s3_bucket, s3_key)

        return {
            "run_date": run_date,
            "rows": len(features),
            "s3_bucket": s3_bucket,
            "s3_key": s3_key,
        }

    @task(task_id="validate_saved_result")
    def validate_saved_result(result_info: dict[str, str | int]) -> None:
    
        if int(result_info["rows"]) <= 0:   #проверка, что файл не пустой
            raise ValueError("Saved feature file is empty")

        s3_client, _ = get_s3_client_and_bucket()
        s3_bucket = str(result_info["s3_bucket"])
        s3_key = str(result_info["s3_key"])

        try:
            metadata = s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
        except ClientError as error:
            error_code = error.response.get("Error", {}).get("Code")
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(
                    f"S3 object was not found: s3://{s3_bucket}/{s3_key}"
                ) from error
            raise

        if int(metadata.get("ContentLength", 0)) <= 0:
            raise ValueError(f"S3 object is empty: s3://{s3_bucket}/{s3_key}")

    validate_saved_result(build_and_upload_features())