"""
Twitch Recommendation Pipeline DAG
Runs daily to recompute Jaccard + Cosine similarity scores across all users,
refresh Neo4j graph edges, validate top-K recommendations, and log results.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import psycopg2
import os
import json
import logging
from neo4j import GraphDatabase

# --- Default DAG args ---
default_args = {
    "owner": "keertana",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# --- DB helpers ---
def get_postgres_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "twitch_recommendations"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "password")
    )

def get_neo4j_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
    )

# --- Task functions ---
def ingest_data(**context):
    """Pull latest user interaction data from PostgreSQL."""
    conn = get_postgres_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM user_interactions")
        count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT user_id) FROM user_interactions")
        unique_users = cur.fetchone()[0]
    conn.close()
    logging.info(f"Ingested {count} interactions across {unique_users} unique users")
    context["ti"].xcom_push(key="interaction_count", value=count)
    context["ti"].xcom_push(key="unique_users", value=unique_users)

def compute_similarity(**context):
    """Recompute Jaccard and Cosine similarity scores between users."""
    conn = get_postgres_conn()

    with conn.cursor() as cur:
        # Jaccard similarity: intersection / union of interacted users
        cur.execute("""
            WITH user_sets AS (
                SELECT user_id, array_agg(DISTINCT target_user_id) AS targets
                FROM user_interactions
                WHERE interaction_score > 0
                GROUP BY user_id
            )
            INSERT INTO similarity_scores (user_a, user_b, jaccard_score, computed_at)
            SELECT
                a.user_id,
                b.user_id,
                CAST(
                    array_length(ARRAY(
                        SELECT unnest(a.targets) INTERSECT SELECT unnest(b.targets)
                    ), 1) AS FLOAT
                ) /
                NULLIF(array_length(ARRAY(
                    SELECT unnest(a.targets) UNION SELECT unnest(b.targets)
                ), 1), 0) AS jaccard_score,
                NOW()
            FROM user_sets a
            CROSS JOIN user_sets b
            WHERE a.user_id < b.user_id
            ON CONFLICT (user_a, user_b)
            DO UPDATE SET jaccard_score = EXCLUDED.jaccard_score, computed_at = NOW()
        """)
    conn.commit()
    conn.close()
    logging.info("Similarity scores recomputed successfully")

def refresh_neo4j(**context):
    """Push updated similarity scores into Neo4j as weighted edges."""
    conn = get_postgres_conn()
    driver = get_neo4j_driver()

    with conn.cursor() as cur:
        cur.execute("""
            SELECT user_a, user_b, jaccard_score
            FROM similarity_scores
            WHERE jaccard_score > 0.1
            ORDER BY jaccard_score DESC
            LIMIT 10000
        """)
        rows = cur.fetchall()

    with driver.session() as session:
        for user_a, user_b, score in rows:
            session.run("""
                MERGE (a:User {id: $user_a})
                MERGE (b:User {id: $user_b})
                MERGE (a)-[r:SIMILAR_TO]->(b)
                SET r.jaccard_score = $score, r.updated = timestamp()
            """, user_a=user_a, user_b=user_b, score=score)

    conn.close()
    driver.close()
    logging.info(f"Refreshed {len(rows)} similarity edges in Neo4j")
    context["ti"].xcom_push(key="edges_refreshed", value=len(rows))

def validate_recommendations(**context):
    """Spot-check top-K recommendations for a sample of users."""
    driver = get_neo4j_driver()
    sample_users = [0, 100, 500, 1000, 2000]
    results = {}

    with driver.session() as session:
        for user_id in sample_users:
            result = session.run("""
                MATCH (u:User {id: $user_id})-[r:SIMILAR_TO]->(rec:User)
                RETURN rec.id AS recommended_user, r.jaccard_score AS score
                ORDER BY r.jaccard_score DESC
                LIMIT 5
            """, user_id=user_id)
            recommendations = [{"user": r["recommended_user"], "score": r["score"]} for r in result]
            results[user_id] = recommendations

    driver.close()
    logging.info(f"Validation results: {json.dumps(results, indent=2)}")
    context["ti"].xcom_push(key="validation_results", value=results)

def log_run_summary(**context):
    """Write pipeline run summary to PostgreSQL for monitoring."""
    ti = context["ti"]
    interaction_count = ti.xcom_pull(key="interaction_count", task_ids="ingest_data")
    unique_users = ti.xcom_pull(key="unique_users", task_ids="ingest_data")
    edges_refreshed = ti.xcom_pull(key="edges_refreshed", task_ids="refresh_neo4j")

    conn = get_postgres_conn()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_run_log (
                run_id SERIAL PRIMARY KEY,
                run_date TIMESTAMP DEFAULT NOW(),
                interaction_count INTEGER,
                unique_users INTEGER,
                edges_refreshed INTEGER
            )
        """)
        cur.execute("""
            INSERT INTO pipeline_run_log (interaction_count, unique_users, edges_refreshed)
            VALUES (%s, %s, %s)
        """, (interaction_count, unique_users, edges_refreshed))
    conn.commit()
    conn.close()
    logging.info(f"Pipeline run logged: {interaction_count} interactions, {unique_users} users, {edges_refreshed} edges")

# --- DAG definition ---
with DAG(
    "twitch_recommendation_pipeline",
    default_args=default_args,
    description="Daily batch pipeline to recompute Twitch user similarity scores and refresh recommendations",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["recommendations", "neo4j", "postgresql"]
) as dag:

    t1 = PythonOperator(task_id="ingest_data", python_callable=ingest_data)
    t2 = PythonOperator(task_id="compute_similarity", python_callable=compute_similarity)
    t3 = PythonOperator(task_id="refresh_neo4j", python_callable=refresh_neo4j)
    t4 = PythonOperator(task_id="validate_recommendations", python_callable=validate_recommendations)
    t5 = PythonOperator(task_id="log_run_summary", python_callable=log_run_summary)

    t1 >> t2 >> t3 >> t4 >> t5
