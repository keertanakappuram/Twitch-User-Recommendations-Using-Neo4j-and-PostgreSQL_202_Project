"""
scoring.py — Similarity-Based Twitch Recommendation System
============================================================
Layer 1 of the recommendation pipeline.

Computes user recommendations using three approaches:
1. Content-Based Filtering  — Cosine similarity on normalized user features (PostgreSQL)
2. Collaborative Filtering  — Graph-based Jaccard similarity via Neo4j GDS
3. Matrix Factorization     — SVD on implicit follow graph

Evaluated with Hit Rate@K, Precision@K, Recall@K, NDCG@K.
Results saved back to PostgreSQL for downstream use by the ML pipeline.

Usage:
    python scoring.py
"""

import pandas as pd
import numpy as np
import psycopg2
import warnings
import json
import time
import os
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from neo4j import GraphDatabase

warnings.simplefilter(action="ignore", category=UserWarning)

# ─────────────────────────────────────────────
# CONFIG — update credentials or use .env
# ─────────────────────────────────────────────
POSTGRES_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "dbname":   os.getenv("POSTGRES_DB",   "twitch_recommendations"),
    "user":     os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "port":     os.getenv("POSTGRES_PORT", "5433")
}
NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER",    "neo4j"),
              os.getenv("NEO4J_PASSWORD", "postgres"))

TOP_K        = 10
N_FACTORS    = 50
TEST_SIZE    = 0.2
SAMPLE_USERS = 200

# ─────────────────────────────────────────────
# DATABASE CONNECTIONS
# ─────────────────────────────────────────────
def get_pg_conn():
    return psycopg2.connect(**POSTGRES_CONFIG)

def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

# ─────────────────────────────────────────────
# 1. LOAD DATA FROM POSTGRESQL
# ─────────────────────────────────────────────
def load_users():
    conn = get_pg_conn()
    query = """
        SELECT u.new_id, u.views, CAST(u.partner AS INT) AS partner,
               u.days, CAST(u.mature AS INT) AS mature, uf.features
        FROM users u
        JOIN user_features uf ON u.new_id = uf.new_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def load_edges_from_csv(filepath="musae_ENGB_edges.csv"):
    import csv
    edges = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append((int(row["from"]), int(row["to"])))
    return edges

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_feature_matrix(users_df):
    print("Building normalized feature matrix...")
    users_df = users_df.copy()

    def parse_features(f):
        if isinstance(f, str):
            return json.loads(f)
        return f if f else []

    users_df["features_parsed"] = users_df["features"].apply(parse_features)
    feature_df = pd.DataFrame(
        users_df["features_parsed"].tolist(),
        index=users_df["new_id"]
    ).fillna(0).astype(float)
    feature_df.columns = [f"feat_{i}" for i in feature_df.columns]

    scaler = MinMaxScaler()
    users_df = users_df.set_index("new_id")
    users_df[["views", "days"]] = scaler.fit_transform(users_df[["views", "days"]])

    base = users_df[["views", "partner", "days", "mature"]]
    full = pd.concat([base, feature_df], axis=1).fillna(0)
    print(f"Feature matrix: {full.shape}")
    return full

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
def split_edges(edges):
    print("Splitting edges into train/test sets...")
    user_edges = {}
    for src, dst in edges:
        user_edges.setdefault(src, []).append(dst)

    train_edges, test_edges = [], []
    for user, targets in user_edges.items():
        if len(targets) < 2:
            train_edges.extend([(user, t) for t in targets])
            continue
        train_t, test_t = train_test_split(targets, test_size=TEST_SIZE, random_state=42)
        train_edges.extend([(user, t) for t in train_t])
        test_edges.extend([(user, t) for t in test_t])

    print(f"Train edges: {len(train_edges):,} | Test edges: {len(test_edges):,}")
    return train_edges, test_edges

# ─────────────────────────────────────────────
# 4. CONTENT-BASED (COSINE)
# ─────────────────────────────────────────────
def compute_content_similarity(feature_matrix):
    print("Computing cosine similarity...")
    sim = cosine_similarity(feature_matrix.values)
    return pd.DataFrame(sim, index=feature_matrix.index, columns=feature_matrix.index)

def get_content_recs(user_id, sim_df, train_follows, k=TOP_K):
    if user_id not in sim_df.index:
        return []
    already = set(train_follows.get(user_id, []))
    scores = sim_df.loc[user_id].drop(index=user_id, errors="ignore")
    scores = scores[~scores.index.isin(already)]
    return list(scores.nlargest(k).index)

# ─────────────────────────────────────────────
# 5. COLLABORATIVE FILTERING — Neo4j GDS
# ─────────────────────────────────────────────
def get_neo4j_jaccard_recs(user_id, driver, train_follows, k=TOP_K):
    """Use Neo4j GDS nodeSimilarity for graph-based Jaccard recommendations."""
    already = set(train_follows.get(user_id, []))
    query = """
        CALL gds.nodeSimilarity.stream('userGraph')
        YIELD node1, node2, similarity
        WITH gds.util.asNode(node1) AS u1, gds.util.asNode(node2) AS u2, similarity
        WHERE u1.new_id = $user_id
        AND NOT EXISTS { MATCH (u1)-[:FOLLOWS]->(u2) }
        RETURN u2.new_id AS recommended_user, similarity
        ORDER BY similarity DESC
        LIMIT $k
    """
    with driver.session() as session:
        result = session.run(query, user_id=user_id, k=k)
        recs = [r["recommended_user"] for r in result
                if r["recommended_user"] not in already]
    return recs[:k]

# ─────────────────────────────────────────────
# 6. MATRIX FACTORIZATION (SVD)
# ─────────────────────────────────────────────
def train_svd(train_edges, user_ids):
    print(f"Training SVD with {N_FACTORS} factors...")
    user_idx = {uid: i for i, uid in enumerate(user_ids)}
    n = len(user_ids)
    rows, cols, data = [], [], []
    for src, dst in train_edges:
        if src in user_idx and dst in user_idx:
            rows.append(user_idx[src])
            cols.append(user_idx[dst])
            data.append(1.0)
    mat = csr_matrix((data, (rows, cols)), shape=(n, n))
    k = min(N_FACTORS, min(mat.shape) - 1)
    U, sigma, Vt = svds(mat.astype(float), k=k)
    predicted = np.dot(np.dot(U, np.diag(sigma)), Vt)
    return predicted, user_idx

def get_svd_recs(user_id, predicted, user_idx, user_ids, train_follows, k=TOP_K):
    if user_id not in user_idx:
        return []
    already = set(train_follows.get(user_id, []))
    scores = predicted[user_idx[user_id]]
    recs = []
    for i in np.argsort(scores)[::-1]:
        candidate = user_ids[i]
        if candidate != user_id and candidate not in already:
            recs.append(candidate)
        if len(recs) == k:
            break
    return recs

# ─────────────────────────────────────────────
# 7. HYBRID (Content + SVD, best alpha=0.0)
# ─────────────────────────────────────────────
def get_hybrid_recs(user_id, sim_df, predicted, user_idx, user_ids,
                    train_follows, k=TOP_K, alpha=0.0):
    already = set(train_follows.get(user_id, []))

    if user_id in sim_df.index:
        cs = sim_df.loc[user_id].drop(index=user_id, errors="ignore")
        cs = (cs - cs.min()) / (cs.max() - cs.min() + 1e-9)
    else:
        cs = pd.Series(dtype=float)

    if user_id in user_idx:
        svd_raw = predicted[user_idx[user_id]]
        svd_s = pd.Series(svd_raw, index=user_ids)
        svd_s = (svd_s - svd_s.min()) / (svd_s.max() - svd_s.min() + 1e-9)
    else:
        svd_s = pd.Series(dtype=float)

    if cs.empty and svd_s.empty:
        return []

    combined = alpha * cs + (1 - alpha) * svd_s.reindex(cs.index, fill_value=0)
    combined = combined[~combined.index.isin(already)].drop(index=user_id, errors="ignore")
    return list(combined.nlargest(k).index)

# ─────────────────────────────────────────────
# 8. EVALUATION METRICS
# ─────────────────────────────────────────────
def precision_at_k(rec, rel, k):
    return len(set(rec[:k]) & set(rel)) / k if k > 0 else 0.0

def recall_at_k(rec, rel, k):
    return len(set(rec[:k]) & set(rel)) / len(rel) if rel else 0.0

def ndcg_at_k(rec, rel, k):
    rel_set = set(rel)
    dcg  = sum(1.0 / np.log2(i + 2) for i, x in enumerate(rec[:k]) if x in rel_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(rel))))
    return dcg / idcg if idcg > 0 else 0.0

def hit_rate_at_k(rec, rel, k):
    return 1.0 if set(rec[:k]) & set(rel) else 0.0

def evaluate(test_edges, sim_df, predicted, user_idx, user_ids,
             train_follows, k=TOP_K, sample=SAMPLE_USERS):
    print(f"\nEvaluating on {sample} sampled users with K={k}...")
    test_gt = {}
    for src, dst in test_edges:
        test_gt.setdefault(src, []).append(dst)

    eval_users = list(test_gt.keys())
    if len(eval_users) > sample:
        np.random.seed(42)
        eval_users = list(np.random.choice(eval_users, sample, replace=False))

    p_list, r_list, n_list, h_list = [], [], [], []
    for uid in eval_users:
        rel = test_gt[uid]
        rec = get_hybrid_recs(uid, sim_df, predicted, user_idx, user_ids, train_follows, k=k)
        p_list.append(precision_at_k(rec, rel, k))
        r_list.append(recall_at_k(rec, rel, k))
        n_list.append(ndcg_at_k(rec, rel, k))
        h_list.append(hit_rate_at_k(rec, rel, k))

    return {
        f"Precision@{k}":  np.mean(p_list),
        f"Recall@{k}":     np.mean(r_list),
        f"NDCG@{k}":       np.mean(n_list),
        f"HitRate@{k}":    np.mean(h_list),
        "Users evaluated": len(eval_users)
    }

# ─────────────────────────────────────────────
# 9. SAVE RESULTS TO POSTGRESQL
# ─────────────────────────────────────────────
def save_similarity_recommendations(recs_dict):
    """Save hybrid similarity recommendations to PostgreSQL."""
    conn = get_pg_conn()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendations_similarity (
            user_id INT, recommended_user INT, rank INT,
            PRIMARY KEY (user_id, recommended_user)
        )
    """)
    cursor.execute("TRUNCATE TABLE recommendations_similarity")
    rows = []
    for user_id, recs in recs_dict.items():
        for rank, rec_user in enumerate(recs, start=1):
            rows.append((user_id, rec_user, rank))
    cursor.executemany("""
        INSERT INTO recommendations_similarity (user_id, recommended_user, rank)
        VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
    """, rows)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved {len(rows)} similarity recommendations to PostgreSQL")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    start = time.time()
    print("=" * 55)
    print("  Twitch Similarity-Based Recommendation System")
    print("=" * 55)

    print("\n[1/6] Loading data from PostgreSQL...")
    users_df = load_users()
    edges    = load_edges_from_csv()
    print(f"Users: {len(users_df):,} | Edges: {len(edges):,}")

    print("\n[2/6] Building feature matrix...")
    feature_matrix = build_feature_matrix(users_df)
    user_ids = list(feature_matrix.index)

    print("\n[3/6] Splitting train/test edges...")
    train_edges, test_edges = split_edges(edges)
    train_follows = {}
    for src, dst in train_edges:
        train_follows.setdefault(src, []).append(dst)

    print("\n[4/6] Computing content-based cosine similarity...")
    sim_df = compute_content_similarity(feature_matrix)

    print("\n[5/6] Training SVD matrix factorization...")
    predicted, user_idx = train_svd(train_edges, user_ids)

    print("\n[6/6] Evaluating hybrid recommendations (best alpha=0.0)...")
    results = evaluate(test_edges, sim_df, predicted, user_idx, user_ids, train_follows)

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    print("=" * 55)

    print("\nGenerating and saving recommendations for all users...")
    all_recs = {}
    for uid in user_ids:
        recs = get_hybrid_recs(uid, sim_df, predicted, user_idx, user_ids, train_follows)
        if recs:
            all_recs[uid] = recs
    save_similarity_recommendations(all_recs)

    print(f"\nDone in {time.time()-start:.1f}s")
    print(f"Recommendations generated for {len(all_recs):,} users")
    print("\nNext step: run ml_link_prediction.py to train ML models on top of these results")

if __name__ == "__main__":
    main()
