"""
Twitch Activity Event Producer
Simulates real-time Twitch user activity events and publishes them to Kafka.
Events include: follow, watch, game_play between users in the MUSAE dataset.
"""

import json
import time
import random
import csv
from kafka import KafkaProducer

# Load existing users from dataset
def load_users(filepath="data/musae_ENGB_target.csv"):
    user_ids = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_ids.append(int(row["new_id"]))
    return user_ids

def generate_event(user_ids):
    """Simulate a Twitch user activity event."""
    action = random.choice(["follow", "watch", "game_play", "unfollow"])
    user = random.choice(user_ids)
    target = random.choice([u for u in user_ids if u != user])
    return {
        "user_id": user,
        "action": action,
        "target_user_id": target,
        "timestamp": time.time(),
        "session_duration": random.randint(30, 7200) if action == "watch" else None
    }

def main():
    producer = KafkaProducer(
        bootstrap_servers=["kafka:9092"],
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    print("Loading users from dataset...")
    user_ids = load_users()
    print(f"Loaded {len(user_ids)} users. Starting event stream...")

    events_sent = 0
    while True:
        event = generate_event(user_ids)
        producer.send("twitch-events", event)
        events_sent += 1
        if events_sent % 100 == 0:
            print(f"[Producer] Sent {events_sent} events. Latest: {event}")
        time.sleep(0.1)  # 10 events/second

if __name__ == "__main__":
    main()
