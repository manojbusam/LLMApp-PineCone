from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging
import requests

# Import your existing functions
from 1_Vector_Database_Indexing import index_documents
from 2_Querying_FastAPI import query_index
from 3_RAGAS_Retrieval_Generation_Evaluation import evaluate_rag

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'vector_database_pipeline',
    default_args=default_args,
    description='A DAG to automate vector database indexing, querying, and evaluation',
    schedule_interval=timedelta(days=1),
    catchup=False
)

def list_s3_files(**kwargs):
    """List files in the S3 bucket"""
    s3_hook = S3Hook(aws_conn_id='aws_default')
    bucket_name = 'your-bucket-name'
    prefix = 'your-prefix/'
    file_list = s3_hook.list_keys(bucket_name=bucket_name, prefix=prefix)
    return file_list

def process_files(**kwargs):
    """Process all files from S3"""
    ti = kwargs['ti']
    file_list = ti.xcom_pull(task_ids='list_s3_files')
    for file_key in file_list:
        index_documents(file_key)

def perform_queries(**kwargs):
    """Perform sample queries"""
    sample_queries = [
        "What are the main advantages of vector databases?",
        "How do vector databases improve search capabilities?",
        "What are some applications of vector databases in AI?",
    ]
    results = [query_index(query) for query in sample_queries]
    return results

def evaluate_results(**kwargs):
    """Evaluate the RAG system"""
    ti = kwargs['ti']
    query_results = ti.xcom_pull(task_ids='perform_queries')
    evaluation_results = evaluate_rag(query_results)
    return evaluation_results

# Define tasks
list_files_task = PythonOperator(
    task_id='list_s3_files',
    python_callable=list_s3_files,
    dag=dag,
)

indexing_task = PythonOperator(
    task_id='index_documents',
    python_callable=process_files,
    dag=dag,
)

query_task = PythonOperator(
    task_id='perform_queries',
    python_callable=perform_queries,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate_rag',
    python_callable=evaluate_results,
    dag=dag,
)

# Set task dependencies
list_files_task >> indexing_task >> query_task >> evaluate_task
