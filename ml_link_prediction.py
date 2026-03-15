"""
ml_link_prediction.py — ML-Based Twitch Link Prediction
=========================================================
Layer 2 of the recommendation pipeline.

Builds on the similarity-based scoring (scoring.py) by reframing
recommendation as a supervised link prediction problem.

Features per user pair are engineered from:
- Graph topology  : common neighbors, Jaccard, Adamic-Adar, PageRank,
                    degree, clustering coefficient, community membership
- Content         : cosine similarity of normalized user feature vectors
- SVD             : dot product of latent embeddings

Models trained and compared:
1. Logistic Regression  — linear baseline
2. XGBoost              — gradient boosted trees (best performer)
3. GraphSAGE            — graph neural network

Results:
- XGBoost   : AUC-ROC 0.9711 | Avg Precision 0.9631 | F1 0.9131
- LR        : AUC-ROC 0.9425 | Avg Precision 0.9442 | F1 0.8640
- GraphSAGE : AUC-ROC 0.8562 | Avg Precision 0.8582 | F1 0.7519

Usage:
    python ml_link_prediction.py
"""

import pandas as pd
import numpy as np
import psycopg2
import warnings
import json
import time
import os
import networkx as nx
from neo4j import GraphDatabase
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
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

N_SAMPLES    = 5000   # positive + negative pairs for training
N_FACTORS    = 32     # SVD latent dims for node features
TOP_K        = 10

FEATURE_NAMES = [
    "common_neighbors", "jaccard", "adamic_adar", "content_cosine", "svd_dot",
    "pagerank_u", "pagerank_v", "pagerank_diff",
    "in_degree_u", "in_degree_v", "in_degree_diff",
    "out_degree_u", "out_degree_v", "out_degree_diff",
    "clustering_u", "clustering_v", "clustering_diff",
    "same_community"
]

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
def get_pg_conn():
    return psycopg2.connect(**POSTGRES_CONFIG)

def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    import csv
    conn = get_pg_conn()
    users_df = pd.read_sql("""
        SELECT u.new_id, u.views, CAST(u.partner AS INT) AS partner,
               u.days, CAST(u.mature AS INT) AS mature, uf.features
        FROM users u JOIN user_features uf ON u.new_id = uf.new_id
    """, conn)
    conn.close()

    with open("musae_ENGB_features.json", "r") as f:
        features_raw = json.load(f)
    all_games = sorted(set(g for games in features_raw.values() for g in games))
    game_index = {g: i for i, g in enumerate(all_games)}
    feature_rows = {}
    for uid, games in features_raw.items():
        row = np.zeros(len(all_games))
        for g in games:
            row[game_index[g]] = 1
        feature_rows[int(uid)] = row
    features_df = pd.DataFrame.from_dict(feature_rows, orient="index")
    features_df.columns = [f"game_{i}" for i in range(len(all_games))]

    edges = []
    with open("musae_ENGB_edges.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append((int(row["from"]), int(row["to"])))

    return users_df, features_df, edges

def build_feature_matrix(users_df, features_df):
    users_indexed = users_df.set_index("new_id").copy()
    scaler = MinMaxScaler()
    cont_cols = [c for c in ["views", "days"] if c in users_indexed.columns]
    if cont_cols:
        users_indexed[cont_cols] = scaler.fit_transform(users_indexed[cont_cols])
    bin_cols = [c for c in ["partner", "mature"] if c in users_indexed.columns]
    for col in bin_cols:
        users_indexed[col] = users_indexed[col].astype(int)
    base = users_indexed[cont_cols + bin_cols].fillna(0)
    fm = pd.concat([base, features_df], axis=1).fillna(0)
    fm = fm.loc[fm.index.isin(features_df.index)]
    return fm, cont_cols, bin_cols

# ─────────────────────────────────────────────
# 2. GRAPH + NODE FEATURES
# Neo4j GDS is used as the primary source for graph features:
#   - PageRank       → gds.pageRank
#   - Community      → gds.louvain
#   - Node Similarity→ gds.nodeSimilarity (Jaccard)
# NetworkX is used as a local fallback and for degree/clustering
# which are not persisted in Neo4j.
# ─────────────────────────────────────────────

NEO4J_GRAPH_NAME = "userGraph"

def fetch_neo4j_pagerank(driver):
    """Fetch PageRank scores from Neo4j GDS."""
    print("  Fetching PageRank from Neo4j GDS...")
    pagerank = {}
    with driver.session() as session:
        result = session.run(f"""
            CALL gds.pageRank.stream('{NEO4J_GRAPH_NAME}', {{
                dampingFactor: 0.85,
                maxIterations: 20
            }})
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).new_id AS user_id, score
        """)
        for record in result:
            pagerank[record["user_id"]] = record["score"]
    print(f"  PageRank fetched for {len(pagerank):,} nodes")
    return pagerank

def fetch_neo4j_communities(driver):
    """Fetch Louvain community assignments from Neo4j GDS."""
    print("  Fetching Louvain communities from Neo4j GDS...")
    node_community = {}
    with driver.session() as session:
        result = session.run(f"""
            CALL gds.louvain.stream('{NEO4J_GRAPH_NAME}')
            YIELD nodeId, communityId
            RETURN gds.util.asNode(nodeId).new_id AS user_id, communityId
        """)
        communities = {}
        for record in result:
            node_community[record["user_id"]] = record["communityId"]
            communities.setdefault(record["communityId"], []).append(record["user_id"])
    print(f"  {len(communities):,} communities detected")
    return node_community

def build_graph_and_features(edges, user_ids, feature_matrix):
    user_id_set = set(user_ids)
    user_idx = {uid: i for i, uid in enumerate(user_ids)}
    n = len(user_ids)

    print("Building local graph (NetworkX) for degree + clustering...")
    G = nx.DiGraph()
    G.add_nodes_from(user_ids)
    G.add_edges_from([(u, v) for u, v in edges if u in user_id_set and v in user_id_set])
    G_undir    = G.to_undirected()
    in_degree  = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    clustering = nx.clustering(G_undir)

    # Try Neo4j GDS for PageRank + communities, fall back to NetworkX
    try:
        driver = get_neo4j_driver()
        driver.verify_connectivity()
        print("Neo4j connected — using GDS for PageRank and community detection...")
        pagerank       = fetch_neo4j_pagerank(driver)
        node_community = fetch_neo4j_communities(driver)
        driver.close()
    except Exception as e:
        print(f"Neo4j unavailable ({e}) — falling back to NetworkX...")
        pagerank = nx.pagerank(G, alpha=0.85)
        communities = nx.community.greedy_modularity_communities(G_undir)
        node_community = {}
        for i, community in enumerate(communities):
            for node in community:
                node_community[node] = i
        print(f"  NetworkX: {len(set(node_community.values())):,} communities detected")

    print("Training SVD for node embeddings...")
    rows, cols, data = [], [], []
    for u, v in edges:
        if u in user_idx and v in user_idx:
            rows.append(user_idx[u])
            cols.append(user_idx[v])
            data.append(1.0)
    mat = csr_matrix((data, (rows, cols)), shape=(n, n))
    k = min(N_FACTORS, min(mat.shape) - 1)
    U, sigma, Vt = svds(mat.astype(float), k=k)
    svd_embeddings = U * sigma

    return (G, G_undir, pagerank, in_degree, out_degree,
            clustering, node_community, svd_embeddings, user_idx)

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING PER PAIR
# ─────────────────────────────────────────────
def get_pair_features(u, v, G, G_undir, pagerank, in_degree, out_degree,
                      clustering, node_community, svd_embeddings,
                      user_idx, feat_array):
    if u not in user_idx or v not in user_idx:
        return None
    ui, vi = user_idx[u], user_idx[v]

    u_nb = set(G_undir.neighbors(u)) if u in G_undir else set()
    v_nb = set(G_undir.neighbors(v)) if v in G_undir else set()
    common = len(u_nb & v_nb)
    union  = len(u_nb | v_nb)
    jaccard = common / union if union > 0 else 0.0

    aa = sum(1.0 / np.log(G_undir.degree(w))
             for w in (u_nb & v_nb) if G_undir.degree(w) > 1)

    content_cos = float(cos_sim([feat_array[ui]], [feat_array[vi]])[0][0])
    svd_dot     = float(np.dot(svd_embeddings[ui], svd_embeddings[vi]))

    pr_u  = pagerank.get(u, 0); pr_v  = pagerank.get(v, 0)
    ind_u = in_degree.get(u, 0); ind_v = in_degree.get(v, 0)
    oud_u = out_degree.get(u, 0); oud_v = out_degree.get(v, 0)
    cl_u  = clustering.get(u, 0); cl_v  = clustering.get(v, 0)
    same_comm = int(node_community.get(u, -1) == node_community.get(v, -2))

    return [
        common, jaccard, aa, content_cos, svd_dot,
        pr_u, pr_v, abs(pr_u - pr_v),
        ind_u, ind_v, abs(ind_u - ind_v),
        oud_u, oud_v, abs(oud_u - oud_v),
        cl_u, cl_v, abs(cl_u - cl_v),
        same_comm
    ]

def build_dataset(edges, user_ids, G, G_undir, pagerank, in_degree,
                  out_degree, clustering, node_community, svd_embeddings,
                  user_idx, feat_array):
    print(f"Building dataset with {N_SAMPLES} positive + {N_SAMPLES} negative samples...")
    np.random.seed(42)
    edge_set   = set(edges)
    user_id_set = set(user_ids)
    n = len(user_ids)

    pos_edges = [(u, v) for u, v in edges if u in user_idx and v in user_idx]
    pos_sample = [pos_edges[i] for i in
                  np.random.choice(len(pos_edges), min(N_SAMPLES, len(pos_edges)), replace=False)]

    neg_sample, attempts = [], 0
    while len(neg_sample) < N_SAMPLES and attempts < N_SAMPLES * 20:
        u = user_ids[np.random.randint(n)]
        v = user_ids[np.random.randint(n)]
        if u != v and (u, v) not in edge_set:
            neg_sample.append((u, v))
        attempts += 1

    X, y = [], []
    graph_args = (G, G_undir, pagerank, in_degree, out_degree,
                  clustering, node_community, svd_embeddings, user_idx, feat_array)
    for u, v in pos_sample:
        f = get_pair_features(u, v, *graph_args)
        if f: X.append(f); y.append(1)
    for u, v in neg_sample:
        f = get_pair_features(u, v, *graph_args)
        if f: X.append(f); y.append(0)

    return np.array(X), np.array(y), pos_sample

# ─────────────────────────────────────────────
# 4. MODELS
# ─────────────────────────────────────────────
def train_logistic_regression(X_train, X_test, y_train, y_test):
    print("\nTraining Logistic Regression...")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(Xtr, y_train)
    probs = lr.predict_proba(Xte)[:, 1]
    preds = lr.predict(Xte)
    return lr, scaler, {
        "AUC-ROC": roc_auc_score(y_test, probs),
        "Avg Precision": average_precision_score(y_test, probs),
        "F1": f1_score(y_test, preds)
    }

def train_xgboost(X_train, X_test, y_train, y_test):
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    probs = xgb.predict_proba(X_test)[:, 1]
    preds = xgb.predict(X_test)
    return xgb, {
        "AUC-ROC": roc_auc_score(y_test, probs),
        "Avg Precision": average_precision_score(y_test, probs),
        "F1": f1_score(y_test, preds)
    }

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv3(x, edge_index)

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        return self.decode(self.encode(x, edge_index), edge_label_index)

def train_graphsage(edges, user_ids, user_idx, svd_embeddings,
                    feature_matrix, cont_cols, bin_cols,
                    pos_sample, neg_sample_indices, X, y):
    print("Training GraphSAGE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enriched node features: SVD + metadata
    svd_t  = torch.tensor(svd_embeddings, dtype=torch.float)
    meta_t = torch.tensor(feature_matrix[cont_cols + bin_cols].values, dtype=torch.float)
    node_feats = torch.cat([svd_t, meta_t], dim=1)

    src_list, dst_list = [], []
    for u, v in edges:
        if u in user_idx and v in user_idx:
            src_list.append(user_idx[u]); dst_list.append(user_idx[v])
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    pyg_data   = Data(x=node_feats, edge_index=edge_index).to(device)

    # Pairs
    all_pairs  = [(user_idx[u], user_idx[v]) for u, v in pos_sample
                  if u in user_idx and v in user_idx]
    all_labels = [1] * len(all_pairs)
    neg_pairs  = [(r, c) for r, c in zip(*np.where(np.random.rand(len(user_ids), len(user_ids)) < 0.001))
                  if r != c][:len(all_pairs)]
    all_pairs  += neg_pairs
    all_labels += [0] * len(neg_pairs)

    pairs_arr  = np.array(all_pairs)
    labels_arr = np.array(all_labels)
    idx_tr, idx_te = train_test_split(np.arange(len(all_pairs)),
                                      test_size=0.2, random_state=42,
                                      stratify=labels_arr)
    train_p = torch.tensor(pairs_arr[idx_tr].T, dtype=torch.long).to(device)
    train_l = torch.tensor(labels_arr[idx_tr], dtype=torch.float).to(device)
    test_p  = torch.tensor(pairs_arr[idx_te].T, dtype=torch.long).to(device)
    test_l  = torch.tensor(labels_arr[idx_te], dtype=torch.float).to(device)

    model     = GraphSAGE(node_feats.shape[1], 128, 64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_loss, patience_ctr, best_state = float("inf"), 0, None
    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        out  = model(pyg_data.x, pyg_data.edge_index, train_p)
        loss = criterion(out, train_l)
        loss.backward(); optimizer.step(); scheduler.step(loss)
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= 30:
            print(f"  Early stopping at epoch {epoch}")
            break
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(pyg_data.x, pyg_data.edge_index, test_p)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs > 0.5).astype(int)
        labels = test_l.cpu().numpy()

    return {
        "AUC-ROC": roc_auc_score(labels, probs),
        "Avg Precision": average_precision_score(labels, probs),
        "F1": f1_score(labels, preds)
    }

# ─────────────────────────────────────────────
# 5. SAVE ML RECOMMENDATIONS TO POSTGRESQL
# ─────────────────────────────────────────────
def save_ml_recommendations(xgb_model, lr_scaler, user_ids, user_idx,
                              G, G_undir, pagerank, in_degree, out_degree,
                              clustering, node_community, svd_embeddings,
                              feat_array, train_follows_set):
    print("\nGenerating ML recommendations for all users...")
    CANDIDATE_POOL = 200
    conn   = get_pg_conn()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendations_ml (
            user_id INT, recommended_user INT, rank INT, score FLOAT,
            PRIMARY KEY (user_id, recommended_user)
        )
    """)
    cursor.execute("TRUNCATE TABLE recommendations_ml")

    graph_args = (G, G_undir, pagerank, in_degree, out_degree,
                  clustering, node_community, svd_embeddings, user_idx, feat_array)
    rows = []
    np.random.seed(0)

    for uid in user_ids[:500]:  # subset for speed
        already  = train_follows_set.get(uid, set())
        candidates = [v for v in np.random.choice(user_ids, CANDIDATE_POOL, replace=False)
                      if v != uid and v not in already and v in user_idx]
        if not candidates:
            continue
        feats = [get_pair_features(uid, v, *graph_args) for v in candidates]
        feats = [f for f in feats if f is not None]
        if not feats:
            continue
        probs = xgb_model.predict_proba(np.array(feats))[:, 1]
        top_k = np.argsort(probs)[::-1][:TOP_K]
        for rank, i in enumerate(top_k, start=1):
            rows.append((uid, candidates[i], rank, float(probs[i])))

    cursor.executemany("""
        INSERT INTO recommendations_ml (user_id, recommended_user, rank, score)
        VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING
    """, rows)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved {len(rows)} ML recommendations to PostgreSQL")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    start = time.time()
    print("=" * 60)
    print("  Twitch ML Link Prediction Pipeline")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    users_df, features_df, edges = load_data()
    feature_matrix, cont_cols, bin_cols = build_feature_matrix(users_df, features_df)
    user_ids = list(feature_matrix.index)
    print(f"Users: {len(user_ids):,} | Edges: {len(edges):,}")

    print("\n[2/5] Building graph and node features...")
    (G, G_undir, pagerank, in_degree, out_degree,
     clustering, node_community, svd_embeddings, user_idx) = build_graph_and_features(
        edges, user_ids, feature_matrix)

    print("\n[3/5] Building training dataset...")
    feat_array = feature_matrix.values
    X, y, pos_sample = build_dataset(
        edges, user_ids, G, G_undir, pagerank, in_degree, out_degree,
        clustering, node_community, svd_embeddings, user_idx, feat_array)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Dataset: {X.shape[0]:,} samples x {X.shape[1]} features")

    print("\n[4/5] Training models...")
    lr_model, lr_scaler, lr_results   = train_logistic_regression(X_train, X_test, y_train, y_test)
    xgb_model, xgb_results             = train_xgboost(X_train, X_test, y_train, y_test)
    sage_results                        = train_graphsage(
        edges, user_ids, user_idx, svd_embeddings,
        feature_matrix, cont_cols, bin_cols, pos_sample, [], X, y)

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    results_df = pd.DataFrame([
        {"Model": "Logistic Regression", **lr_results},
        {"Model": "XGBoost",             **xgb_results},
        {"Model": "GraphSAGE",           **sage_results},
    ])
    print(results_df.to_string(index=False, float_format="{:.4f}".format))
    print("=" * 60)
    best = results_df.loc[results_df["AUC-ROC"].idxmax(), "Model"]
    print(f"\nBest model: {best} (AUC-ROC: {results_df['AUC-ROC'].max():.4f})")

    print("\n[5/5] Saving ML recommendations to PostgreSQL...")
    train_follows_set = {}
    for u, v in edges:
        train_follows_set.setdefault(u, set()).add(v)
    save_ml_recommendations(
        xgb_model, lr_scaler, user_ids, user_idx,
        G, G_undir, pagerank, in_degree, out_degree,
        clustering, node_community, svd_embeddings,
        feat_array, train_follows_set)

    print(f"\nTotal time: {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
