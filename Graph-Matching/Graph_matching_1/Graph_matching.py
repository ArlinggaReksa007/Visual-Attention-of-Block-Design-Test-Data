# =========================================
# IMPORT LIBRARY
# =========================================
from neo4j import GraphDatabase
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ot  # POT: Python Optimal Transport

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS

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
SEED = 42
np.random.seed(SEED)

AOI_LIST = ["Area 1", "Area 2", "Area 3", "Area 4", "Model"]

# Data yang mau dibandingkan
samples = [
    ("Candra", 1),
    ("Candra", 2),
    ("Candra", 3),
    ("Candra", 4),
    ("Arif", 1),
    ("Arif", 2),
    ("Arif", 3),
    ("Arif", 4),
    ("Amar", 1),
    ("Amar", 2),
    ("Amar", 3),
    ("Amar", 4),
    ("Bayu", 1),
    ("Bayu", 2),
    ("Bayu", 3),
    ("Bayu", 4),
    ("genius", 1),
    ("superior", 1),
    ("average", 1),
]

# Graph referensi / kunci
KEY_PARTICIPANT = "Kunci"
KEY_SOAL = 1
KEY_LABEL = "Kunci"

# Balance antara struktur dan fitur
# 0.0 = hanya fitur node
# 1.0 = hanya struktur graph
ALPHA = 0.60


# =========================================
# FUNGSI AMBIL REPRESENTASI GRAPH
# =========================================
def build_graph_representation(participant, soal):

    # node baru: in & out
    AOI_EXT = []
    for aoi in AOI_LIST:
        AOI_EXT.append(aoi + "_out")
        AOI_EXT.append(aoi + "_in")

    # fitur node
    aoi_features = {
        aoi: {"total_time": 0.0, "fixation_count": 0.0}
        for aoi in AOI_LIST
    }

    transition_counts = {}

    with driver.session() as session:

        # LOOKED_AT
        query_looked_at = """
        MATCH (:Participant {name:$participant, soal:$soal})-[r:LOOKED_AT]->(a:AOI {participant:$participant, soal:$soal})
        RETURN a.name AS name,
               coalesce(r.total_time, 0) AS total_time,
               coalesce(r.fixation_count, 0) AS fixation_count
        """

        result = session.run(query_looked_at, participant=participant, soal=soal)

        for record in result:
            name = record["name"]
            if name in aoi_features:
                aoi_features[name]["total_time"] = float(record["total_time"])
                aoi_features[name]["fixation_count"] = float(record["fixation_count"])

        # TRANSITION
        query_transition = """
        MATCH (a:AOI {participant:$participant, soal:$soal})-[t:TRANSITION]->(b:AOI {participant:$participant, soal:$soal})
        RETURN a.name AS from_name,
               b.name AS to_name,
               coalesce(t.count, 0) AS count
        """

        result = session.run(query_transition, participant=participant, soal=soal)

        for record in result:
            a = record["from_name"]
            b = record["to_name"]
            count = float(record["count"])

            if a in AOI_LIST and b in AOI_LIST:
                transition_counts[(a + "_out", b + "_in")] = count

    # =================================
    # FITUR NODE
    # =================================
    X_raw = []

    for aoi in AOI_LIST:
        feat = [
            aoi_features[aoi]["total_time"],
            aoi_features[aoi]["fixation_count"]
        ]
        X_raw.append(feat)  # out
        X_raw.append(feat)  # in

    X_raw = np.array(X_raw, dtype=float)

    # =================================
    # GRAPH STRUCTURE
    # =================================
    G = nx.Graph()
    G.add_nodes_from(AOI_EXT)

    for (u, v), count in transition_counts.items():
        cost = 1.0 / (count + 1.0)
        G.add_edge(u, v, weight=cost)

    n = len(AOI_EXT)
    C = np.full((n, n), np.inf)

    for i, src in enumerate(AOI_EXT):
        lengths = nx.single_source_dijkstra_path_length(G, src, weight="weight")
        for j, dst in enumerate(AOI_EXT):
            if dst in lengths:
                C[i, j] = lengths[dst]

    # handle infinity
    finite = C[np.isfinite(C)]
    if finite.size > 0:
        max_val = finite.max()
        C[~np.isfinite(C)] = max_val * 1.5

    np.fill_diagonal(C, 0.0)

    # normalize
    C = C / (C.max() + 1e-9)

    return C, X_raw


# =========================================
# FUNGSI FGW
# =========================================
def compute_fgw_distance(C1, X1, C2, X2, alpha=ALPHA):
    """
    Menghitung FGW distance antara dua graph.

    alpha:
    - makin mendekati 0 -> lebih fokus ke fitur node
    - makin mendekati 1 -> lebih fokus ke struktur graph
    """

    n1 = X1.shape[0]
    n2 = X2.shape[0]

    p = np.ones(n1) / n1
    q = np.ones(n2) / n2

    # Cost fitur antar node
    M = ot.dist(X1, X2, metric="euclidean")

    # FGW distance
    dist = ot.gromov.fused_gromov_wasserstein2(
        M, C1, C2, p, q, alpha=alpha
    )

    return float(dist)


# =========================================
# BUILD SEMUA GRAPH
# =========================================
items = samples + [(KEY_PARTICIPANT, KEY_SOAL)]

labels = []
raw_graphs = []

print("=== LOADING GRAPH DATA ===")

for participant, soal in items:
    C_raw, X_raw = build_graph_representation(participant, soal)
    raw_graphs.append({
        "participant": participant,
        "soal": soal,
        "C_raw": C_raw,
        "X_raw": X_raw
    })

    if participant == KEY_PARTICIPANT and soal == KEY_SOAL:
        labels.append(KEY_LABEL)
    else:
        labels.append(f"{participant}\nSoal {soal}")


# =========================================
# STANDARDIZE FITUR NODE SECARA GLOBAL
# =========================================
# Supaya total_time dan fixation_count punya skala yang konsisten
all_X = np.vstack([g["X_raw"] for g in raw_graphs])

scaler = StandardScaler()
scaler.fit(all_X)

for g in raw_graphs:
    g["X"] = scaler.transform(g["X_raw"])
    g["C"] = g["C_raw"]


# =========================================
# HITUNG DISTANCE MATRIX ANTAR SEMUA GRAPH
# =========================================
n_graph = len(raw_graphs)
dist_matrix = np.zeros((n_graph, n_graph), dtype=float)

print("\n=== COMPUTING FGW DISTANCES ===")

for i in range(n_graph):
    for j in range(i, n_graph):
        d = compute_fgw_distance(
            raw_graphs[i]["C"], raw_graphs[i]["X"],
            raw_graphs[j]["C"], raw_graphs[j]["X"],
            alpha=ALPHA
        )
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d


# =========================================
# CONVERT KE SIMILARITY
# =========================================
sim_matrix = 1.0 / (1.0 + dist_matrix)

key_index = labels.index(KEY_LABEL)

print("\n=== SCORE TERHADAP GRAPH KUNCI ===")
ranking = []

for i, label in enumerate(labels):
    if i == key_index:
        continue

    sim = sim_matrix[i, key_index]
    score = sim * 100.0
    ranking.append((label, sim, score))
    clean_label = label.replace("\n", " ")
    print(f"{clean_label} -> similarity: {sim:.4f} | score: {score:.2f}")

ranking.sort(key=lambda x: x[1], reverse=True)


# =========================================
# HEATMAP SIMILARITY
# =========================================
plt.figure(figsize=(10, 8))
plt.imshow(sim_matrix, cmap="magma", aspect="auto")
plt.colorbar(label="Similarity")

plt.xticks(range(n_graph), labels, rotation=45, ha="right")
plt.yticks(range(n_graph), labels)

for i in range(n_graph):
    for j in range(n_graph):
        val = sim_matrix[i, j]
        text_color = "white" if val < 0.995 else "black"
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

plt.title("Similarity Antar Peserta (FGW)")
plt.tight_layout()
plt.show()


# =========================================
# CLUSTERING
# =========================================
# Ubah similarity menjadi distance untuk clustering
cluster_dist = dist_matrix.copy()

# Jumlah cluster bisa lo ubah sesuai kebutuhan
n_clusters = min(3, n_graph)

try:
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average"
    )
except TypeError:
    # Kompatibilitas untuk versi scikit-learn lama
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        linkage="average"
    )

cluster_labels = clustering.fit_predict(cluster_dist)

print("\n=== HASIL CLUSTER ===")
for i, label in enumerate(labels):
    clean_label = label.replace("\n", " ")
    print(f"{clean_label} -> Cluster {cluster_labels[i]}")


# =========================================
# VISUALISASI CLUSTER 2D (MDS)
# =========================================
mds = MDS(
    n_components=2,
    dissimilarity="precomputed",
    random_state=SEED,
    n_init=4
)
coords = mds.fit_transform(cluster_dist)

plt.figure(figsize=(8, 6))

for i, label in enumerate(labels):
    x, y = coords[i]

    if label == KEY_LABEL:
        plt.scatter(x, y, marker="*", s=300)
        plt.annotate(
            KEY_LABEL,
            (x, y),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold"
        )
    else:
        plt.scatter(x, y)
        plt.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            va="top",
            fontsize=9
        )

plt.title("Clustering Pola Atensi Visual (FGW)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================
# TUTUP KONEKSI
# =========================================
driver.close()