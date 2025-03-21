import pandas as pd
import psycopg2
from neo4j import GraphDatabase
import warnings

# Function to get the database connection
def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",  
        dbname="twitch_recommendations", 
        user="postgres",  
        password="postgres",  
        port="5433"  
    )
    return conn
warnings.simplefilter(action='ignore', category=UserWarning)
# Function to compute cosine similarity in SQL
def get_cosine_similarity_recommendations(target_user_id):
    conn = get_db_connection()
    
    query = f"""
WITH user_features_final AS (
    SELECT 
        u.new_id, 
        u.views, 
        CAST(u.partner AS INT) AS partner,  
        u.days, 
        CAST(u.mature AS INT) AS mature,
        CAST(uf.features AS jsonb) AS features  -- Ensure it's jsonb type
    FROM users u
    JOIN user_features uf ON u.new_id = uf.new_id
),
dot_product AS (
    SELECT 
        uf1.new_id AS user_id_1,
        uf2.new_id AS user_id_2,
        (
            CAST(uf1.views AS FLOAT8) * CAST(uf2.views AS FLOAT8) + 
            CAST(uf1.partner AS FLOAT8) * CAST(uf2.partner AS FLOAT8) + 
            CAST(uf1.days AS FLOAT8) * CAST(uf2.days AS FLOAT8) + 
            CAST(uf1.mature AS FLOAT8) * CAST(uf2.mature AS FLOAT8) +
            COALESCE((
                SELECT SUM(f1::FLOAT8 * f2::FLOAT8)
                FROM jsonb_array_elements(uf1.features) WITH ORDINALITY AS t1(f1, ord1)
                JOIN jsonb_array_elements(uf2.features) WITH ORDINALITY AS t2(f2, ord2)
                ON t1.ord1 = t2.ord2
            ), 0)
        ) AS dot_prod  
    FROM user_features_final uf1
    JOIN user_features_final uf2 
        ON uf1.new_id != uf2.new_id  
    WHERE uf1.new_id = {target_user_id}  
),
magnitude AS (
    SELECT 
        new_id,
        SQRT(
            CAST(POW(views, 2) AS FLOAT8) + 
            CAST(POW(partner, 2) AS FLOAT8) + 
            CAST(POW(days, 2) AS FLOAT8) + 
            CAST(POW(mature, 2) AS FLOAT8) +
            COALESCE((
                SELECT SUM(f::FLOAT8 * f::FLOAT8)
                FROM jsonb_array_elements(features) AS t(f)
            ), 0)
        ) AS magnitude  
    FROM user_features_final
)
SELECT 
    dp.user_id_2 AS recommended_user,
    dp.dot_prod / (m1.magnitude * m2.magnitude) AS cosine_similarity
FROM dot_product dp
JOIN magnitude m1 ON dp.user_id_1 = m1.new_id
JOIN magnitude m2 ON dp.user_id_2 = m2.new_id
WHERE dp.user_id_1 != dp.user_id_2  
ORDER BY cosine_similarity DESC;
    """
    try:
        df = pd.read_sql(query, conn) 
        return df
    finally:
        conn.close()

# Function to insert recommendations into the database
def insert_recommendations(user_id, recommendations):
    conn = psycopg2.connect(
        host="localhost",
        dbname="twitch_recommendations",
        user="postgres",
        password="postgres",
        port="5433"
    )
    cursor = conn.cursor()

    # Create recommendations table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recommendations_content (
        user_id INT,
        recommended_user INT,
        cosine_similarity FLOAT8
    );
    """)

    # Prepare data for insertion (convert NumPy floats to Python floats)
    data_to_insert = [(user_id, int(row['recommended_user']), float(row['cosine_similarity'])) for _, row in recommendations.iterrows()]

    insert_query = """
    INSERT INTO recommendations_content (user_id, recommended_user, cosine_similarity)
    VALUES (%s, %s, %s);
    """

    cursor.executemany(insert_query, data_to_insert)
    conn.commit()
    cursor.close()
    conn.close()

# Example usage:
target_user_id = 6194
recommendations = get_cosine_similarity_recommendations(target_user_id)

# Insert recommendations into the database
insert_recommendations(target_user_id, recommendations)


URI = "bolt://localhost:7687"
AUTH = ("neo4j", "postgres")  

driver = GraphDatabase.driver(URI, auth=AUTH)

def fetch_neo4j_results(query, parameters=None):
    with driver.session() as session:
        result = session.run(query, parameters)
        return [dict(record) for record in result]

def get_user_recommendations(user_id, similarity_threshold=0.0):
    query = """
    CALL gds.nodeSimilarity.stream('userGraph')
    YIELD node1, node2, similarity
    WITH gds.util.asNode(node1) AS user1, gds.util.asNode(node2) AS user2, similarity
    WHERE user1.new_id = $user_id AND similarity > $threshold
    AND NOT EXISTS { MATCH (user1)-[:FOLLOWS]->(user2) } 
    RETURN user2.new_id AS recommended_user, similarity
    ORDER BY similarity DESC
    """
    
    parameters = {"user_id": user_id, "threshold": similarity_threshold}
    return fetch_neo4j_results(query, parameters)


def create_recommendations_table():
    # Establish PostgreSQL connection
    conn = psycopg2.connect(
        host="localhost",  # e.g., 'localhost'
        dbname="twitch_recommendations",  # e.g., 'content_recommendations'
        user="postgres",  # e.g., 'your_user'
        password="postgres",  # e.g., 'your_password'
        port="5433"  # default: 5432
    )
    cursor = conn.cursor()
    
    # Create the recommendations table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recommendations_collab (
        user_id INT,
        recommended_user INT,
        similarity FLOAT,
        PRIMARY KEY (user_id, recommended_user)
    );
    """)
    conn.commit()
    cursor.close()
    conn.close()

def insert_recommendations_to_postgres(user_id, recommendations):
    # Establish PostgreSQL connection
    conn = psycopg2.connect(
        host="localhost",  # e.g., 'localhost'
        dbname="twitch_recommendations",  # e.g., 'content_recommendations'
        user="postgres",  # e.g., 'your_user'
        password="postgres",  # e.g., 'your_password'
        port="5433"  # default: 5432
    )
    cursor = conn.cursor()
    
    # Insert recommendations into the recommendations_collab table
    for rec in recommendations:
        cursor.execute("""
        INSERT INTO recommendations_collab (user_id, recommended_user, similarity)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id, recommended_user) DO NOTHING;
        """, (user_id, rec['recommended_user'], rec['similarity']))
    
    conn.commit()
    cursor.close()
    conn.close()

# Create the recommendations table
create_recommendations_table()

# Get recommendations for a user
user_id = 6194
recommendations_collab = get_user_recommendations(user_id)

# Insert recommendations into PostgreSQL table
insert_recommendations_to_postgres(user_id, recommendations_collab)

def get_combined_recommendations():
    conn = psycopg2.connect(
        host="localhost",
        dbname="twitch_recommendations",
        user="postgres",
        password="postgres",
        port="5433"
    )
    
    query = """
SELECT 
    rc.user_id, 
    rc.recommended_user, 
    rc.similarity AS similarity_score_collab, 
    rcon.cosine_similarity AS similarity_score_context, 
    (0.5 * rc.similarity + 0.5 * rcon.cosine_similarity) AS final_score
FROM recommendations_collab rc
JOIN recommendations_content rcon 
ON rc.user_id = rcon.user_id 
AND rc.recommended_user = rcon.recommended_user
GROUP BY rc.user_id, rc.recommended_user, rc.similarity, rcon.cosine_similarity
ORDER BY final_score DESC
LIMIT 5;

    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    return df
# Fetch and print recommendations
final_recommendations = get_combined_recommendations()
print("Following are the recommendations for user",user_id,":")
print(final_recommendations)