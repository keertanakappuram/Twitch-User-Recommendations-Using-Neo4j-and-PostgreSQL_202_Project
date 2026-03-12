# 🎮 Twitch User Recommendations Using Neo4j & PostgreSQL

> A graph-based user recommendation system that computes **Jaccard and Cosine similarity** scores across a Twitch social network to surface the most relevant user recommendations for any target user.

---

## 🎬 Demo

<video src="https://github.com/user-attachments/assets/702429d8-9da4-43a1-9b4c-243274487af5" controls width="100%"></video>

---

## 🎯 Problem

Recommending relevant users on a social platform like Twitch requires understanding both shared interests and network relationships. This project combines a **relational database (PostgreSQL)** for structured user data with a **graph database (Neo4j)** to model social connections — computing similarity scores to recommend the most relevant users to any target user.

---

## 📦 Dataset

This project uses the [Twitch Social Network Dataset (MUSAE)](https://snap.stanford.edu/data/twitch-social-networks.html) from Stanford SNAP — a real-world graph dataset of Twitch user connections and features across different language communities.
---

## 🔍 Approach

### 1. Data Ingestion
- Loaded Twitch user features (JSON) and edge relationships (CSV) into both PostgreSQL and Neo4j
- Created structured tables in PostgreSQL for user metadata and features
- Imported graph edges and nodes into Neo4j for relationship traversal

### 2. Similarity Scoring
- **Jaccard Similarity** — measures overlap in shared game/category preferences between users
- **Cosine Similarity** — measures directional alignment of user feature vectors
- Each target user receives a similarity score against every other user in the network

### 3. Recommendation Generation
- Users with the highest combined similarity scores are surfaced as top recommendations
- Graph Data Science (GDS) library in Neo4j used for efficient graph-based computation

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Graph Database | Neo4j, Cypher, Graph Data Science Library |
| Relational Database | PostgreSQL |
| Data Processing | Python, Pandas |
| Similarity Metrics | Jaccard Similarity, Cosine Similarity |

---

## 📁 Repository Structure

```
├── scoring.py                        # Main similarity scoring and recommendation logic
├── load_json_data.py                 # Loads JSON feature data into PostgreSQL
├── postgresql_table_creation.sql     # SQL schema and table setup
├── neo4j.cypher                      # Cypher queries for Neo4j graph setup and GDS
├── musae_ENGB_edges.csv              # Twitch user social graph edges
├── musae_ENGB_features.json          # Twitch user feature vectors
├── musae_ENGB_target.csv             # User target labels
├── requirements.txt                  # Python dependencies
├── 202 - Final Report.pdf            # Full project report
└── README.md
```

---

## 🚀 How to Run

### Prerequisites
- PostgreSQL installed and running
- Neo4j Desktop installed with the Graph Data Science (GDS) plugin

### Step 1 — PostgreSQL Setup
1. Create a new database called `twitch_recommendations` in PostgreSQL
2. Run `postgresql_table_creation.sql` to create the `users` and `user_features` tables
3. Run `load_json_data.py` to load the dataset into PostgreSQL (update your DB credentials in the script)

### Step 2 — Neo4j Setup
1. Open Neo4j Desktop and create a new project
2. Add a Local DBMS, set a name and password
3. Open the DBMS import folder and copy in: `musae_ENGB_edges.csv`, `musae_ENGB_target.csv`, `musae_ENGB_features.json`
4. In the Plugins tab, install the **Graph Data Science Library**
5. Open Neo4j Browser and run all queries in `neo4j.cypher`

### Step 3 — Run the Recommender
```bash
pip install -r requirements.txt
python scoring.py   # Update PostgreSQL and Neo4j credentials before running
```
