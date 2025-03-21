import psycopg2
import json
import csv

# Database connection
conn = psycopg2.connect("dbname=twitch_recommendations user=postgres password=postgres host=localhost port=5433")
cursor = conn.cursor()

# Load JSON file and insert into user_features table
with open("/Users/keertanakappuram/musae_ENGB_features.json", "r") as f:
    user_features = json.load(f)

for user_id, features in user_features.items():
    cursor.execute("""
        INSERT INTO user_features (new_id, features)
        VALUES (%s, %s)
        ON CONFLICT (new_id) DO NOTHING;
    """, (user_id, json.dumps(features)))

# Load CSV file (comma-separated) and insert into users table
with open("/Users/keertanakappuram/musae_ENGB_target.csv", "r") as f:
    csv_reader = csv.reader(f)  # Default delimiter is ',' (comma)
    next(csv_reader)  # Skip header row
    
    for row in csv_reader:
        user_id, days, mature, views, partner, new_id = row
        cursor.execute("""
            INSERT INTO users (ids, days, mature, views, partner, new_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (new_id) DO NOTHING;
        """, (int(user_id), int(days), mature.lower() == 'true', int(views), partner.lower() == 'true', int(new_id)))

# Commit & close
conn.commit()
cursor.close()
conn.close()
print("Data loaded successfully into user_features and users tables!")