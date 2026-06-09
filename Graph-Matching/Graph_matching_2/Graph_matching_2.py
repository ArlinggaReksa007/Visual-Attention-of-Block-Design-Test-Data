# =========================================
# IMPORT LIBRARY
# =========================================
from neo4j import GraphDatabase
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ot
from sklearn.preprocessing import StandardScaler

# =========================================
# KONFIGURASI NEO4J
# =========================================
URI = "bolt://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "29051999"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# =========================================
# SETTING
# =========================================
np.random.seed(42)

AOI_LIST = ["Area 1", "Area 2", "Area 3", "Area 4", "Model"]

samples = [
    ("Candra", 1), ("Candra", 2), ("Candra", 3), ("Candra", 4),
    ("Arif", 1), ("Arif", 2), ("Arif", 3), ("Arif", 4),
    ("Amar", 1), ("Amar", 2), ("Amar", 3), ("Amar", 4),
    ("Bayu", 1), ("Bayu", 2), ("Bayu", 3), ("Bayu", 4),
    ("genius", 1), ("superior", 1), ("average", 1),
]

KEY = ("Kunci", 1)
ALPHA = 0.60
TOP_K = 3

# =========================================
# UTIL
# =========================================
def clean_label(text):
    return str(text).replace("\n", " ")

# =========================================
# BUILD GRAPH REPRESENTATION
# =========================================
def build_graph_representation(participant, soal):
    aoi_ext = [a + "_out" for a in AOI_LIST] + [a + "_in" for a in AOI_LIST]
    aoi_features = {a: [0.0, 0.0] for a in AOI_LIST}
    transitions = {}

    with driver.session() as session:
        q1 = """
        MATCH (:Participant {name:$p, soal:$s})-[r:LOOKED_AT]->(a:AOI {participant:$p, soal:$s})
        RETURN a.name AS name, r.total_time AS t, r.fixation_count AS f
        """
        for r in session.run(q1, p=participant, s=soal):
            if r["name"] in aoi_features:
                aoi_features[r["name"]] = [float(r["t"] or 0), float(r["f"] or 0)]

        q2 = """
        MATCH (a:AOI {participant:$p, soal:$s})-[t:TRANSITION]->(b:AOI {participant:$p, soal:$s})
        RETURN a.name AS a, b.name AS b, t.count AS c
        """
        for r in session.run(q2, p=participant, s=soal):
            transitions[(r["a"] + "_out", r["b"] + "_in")] = float(r["c"] or 0)

    # fitur node
    X_raw = np.array([feat for a in AOI_LIST for feat in (aoi_features[a], aoi_features[a])], dtype=float)

    # struktur graph
    G = nx.Graph()
    G.add_nodes_from(aoi_ext)
    for (u, v), c in transitions.items():
        G.add_edge(u, v, weight=1.0 / (c + 1.0))

    n = len(aoi_ext)
    C = np.full((n, n), np.inf)

    for i, src in enumerate(aoi_ext):
        dist = nx.single_source_dijkstra_path_length(G, src, weight="weight")
        for j, dst in enumerate(aoi_ext):
            if dst in dist:
                C[i, j] = dist[dst]

    finite = C[np.isfinite(C)]
    if finite.size > 0:
        C[~np.isfinite(C)] = finite.max() * 1.5

    np.fill_diagonal(C, 0.0)
    C = C / (C.max() + 1e-9)

    return C, X_raw

# =========================================
# FGW DISTANCE
# =========================================
def compute_fgw_distance(C1, X1, C2, X2):
    p = np.ones(len(X1)) / len(X1)
    q = np.ones(len(X2)) / len(X2)
    M = ot.dist(X1, X2, metric="euclidean")
    return float(ot.gromov.fused_gromov_wasserstein2(M, C1, C2, p, q, alpha=ALPHA))

# =========================================
# LOAD ALL GRAPHS
# =========================================
items = samples + [KEY]

labels = []
graphs = []

print("=== LOADING GRAPH DATA ===")
for p, s in items:
    C_raw, X_raw = build_graph_representation(p, s)
    graphs.append({"C_raw": C_raw, "X_raw": X_raw})
    labels.append("Kunci" if (p, s) == KEY else f"{p}_S{s}")

# =========================================
# STANDARDIZE FEATURES
# =========================================
all_X = np.vstack([g["X_raw"] for g in graphs])
scaler = StandardScaler().fit(all_X)

for g in graphs:
    g["X"] = scaler.transform(g["X_raw"])
    g["C"] = g["C_raw"]

# =========================================
# DISTANCE & SIMILARITY MATRIX
# =========================================
n = len(graphs)
dist_matrix = np.zeros((n, n), dtype=float)

print("\n=== COMPUTING FGW DISTANCES ===")
for i in range(n):
    for j in range(i, n):
        d = compute_fgw_distance(
            graphs[i]["C"], graphs[i]["X"],
            graphs[j]["C"], graphs[j]["X"]
        )
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d

sim = 1.0 / (1.0 + dist_matrix)
key_idx = labels.index("Kunci")

# =========================================
# PRINT SIMILARITY MATRIX
# =========================================
print("\n=== SIMILARITY MATRIX ===")
for i, lbl in enumerate(labels):
    row = " ".join(f"{sim[i, j]:.2f}" for j in range(n))
    print(f"{lbl:12s}: {row}")

# =========================================
# RANKING KE KUNCI
# =========================================
print("\n=== RANKING KE KUNCI ===")
ranking = []
for i, lbl in enumerate(labels):
    if lbl == "Kunci":
        continue
    score = sim[i, key_idx]
    ranking.append((lbl, score))

ranking.sort(key=lambda x: x[1], reverse=True)

for idx, (lbl, score) in enumerate(ranking, start=1):
    print(f"{idx:2d}. {lbl:12s} -> {score:.3f}")

# =========================================
# NEAREST NEIGHBOR
# =========================================
print("\n=== NEAREST NEIGHBOR ===")
topk_mean = []

for i in range(n):
    sims = sim[i].copy()
    sims[i] = -1  # buang self-similarity

    idx = np.argsort(sims)[-TOP_K:]
    nn_names = [labels[j] for j in idx]
    nn_scores = [sims[j] for j in idx]
    topk_mean.append(np.mean(nn_scores))

    print(f"{labels[i]:12s} -> {nn_names}")

# =========================================
# ANCHOR PLOT
# =========================================
plt.figure(figsize=(10, 7))

for i, lbl in enumerate(labels):
    if lbl == "Kunci":
        x, y = 1.0, 1.0
        plt.scatter(x, y, s=320, marker="*", color="red", edgecolor="black", linewidth=0.8)
        plt.text(x, y, "Kunci", ha="center", va="bottom", fontsize=10, fontweight="bold")
    else:
        x = sim[i, key_idx]
        y = topk_mean[i]
        plt.scatter(x, y, s=80, color="steelblue", edgecolor="black", linewidth=0.6)
        plt.text(x, y, lbl, fontsize=8, ha="left", va="bottom")

plt.xlim(0, 1.03)
plt.ylim(0, 1.03)
plt.xlabel("Similarity ke Kunci")
plt.ylabel(f"Rata-rata similarity ke {TOP_K} tetangga terdekat")
plt.title("Anchor Plot Similarity Pola Atensi Visual (FGW)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

driver.close()