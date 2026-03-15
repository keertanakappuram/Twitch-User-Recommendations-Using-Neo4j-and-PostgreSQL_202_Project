"""
Twitch Activity Event Consumer
Reads user activity events from Kafka, updates similarity scores in PostgreSQL,
and refreshes relationship weights in Neo4j graph.
"""

import json
import os
import psycopg2
from neo4j import GraphDatabase
from kafka import KafkaConsumer
from collections import defaultdict

# --- Database connections ---
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

# --- Score update logic ---
def update_interaction_score(pg_conn, user_id, target_user_id, action):
    """Update interaction count between users in PostgreSQL."""
    action_weights = {
        "follow": 3.0,
        "watch": 2.0,
        "game_play": 1.5,
        "unfollow": -1.0
    }
    weight = action_weights.get(action, 1.0)

    with pg_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO user_interactions (user_id, target_user_id, interaction_score)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, target_user_id)
            DO UPDATE SET interaction_score = user_interactions.interaction_score + EXCLUDED.interaction_score,
                          last_updated = NOW()
        """, (user_id, target_user_id, weight))
    pg_conn.commit()

def update_neo4j_relationship(driver, user_id, target_user_id, action):
    """Update relationship weight in Neo4j based on new activity."""
    weight_delta = {"follow": 3.0, "watch": 2.0, "game_play": 1.5, "unfollow": -1.0}.get(action, 1.0)

    with driver.session() as session:
        session.run("""
            MATCH (a:User {id: $user_id}), (b:User {id: $target_id})
            MERGE (a)-[r:INTERACTED_WITH]->(b)
            ON CREATE SET r.weight = $delta, r.updated = timestamp()
            ON MATCH SET r.weight = r.weight + $delta, r.updated = timestamp()
        """, user_id=user_id, target_id=target_user_id, delta=weight_delta)

# --- Main consumer loop ---
def main():
    print("Connecting to Kafka...")
    consumer = KafkaConsumer(
        "twitch-events",
        bootstrap_servers=["kafka:9092"],
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="twitch-recommender"
    )

    print("Connecting to PostgreSQL and Neo4j...")
    pg_conn = get_postgres_conn()
    neo4j_driver = get_neo4j_driver()

    # Ensure interactions table exists
    with pg_conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                user_id INTEGER,
                target_user_id INTEGER,
                interaction_score FLOAT DEFAULT 0,
                last_updated TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (user_id, target_user_id)
            )
        """)
    pg_conn.commit()

    print("Consuming events from twitch-events topic...")
    events_processed = 0

    for message in consumer:
        event = message.value
        user_id = event["user_id"]
        target_user_id = event["target_user_id"]
        action = event["action"]

        update_interaction_score(pg_conn, user_id, target_user_id, action)
        update_neo4j_relationship(neo4j_driver, user_id, target_user_id, action)

        events_processed += 1
        if events_processed % 50 == 0:
            print(f"[Consumer] Processed {events_processed} events. Latest: user {user_id} -> {action} -> user {target_user_id}")

if __name__ == "__main__":
    main()
