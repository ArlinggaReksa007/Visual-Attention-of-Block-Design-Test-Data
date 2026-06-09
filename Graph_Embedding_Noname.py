# =========================================
# IMPORT
# =========================================
from neo4j import GraphDatabase
import networkx as nx
from karateclub import Graph2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# =========================================
# PLOT STYLE
# =========================================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# =========================================
# NEO4J CONFIG
# =========================================
URI = "bolt://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "29051999"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# =========================================
# DATA SAMPLE
# =========================================
samples = [
    ("Kunci", 1),

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

KEY_LABEL = "key"

# =========================================
# DISPLAY NAME MAPPING
# =========================================
special_names = {
    "Kunci": "key",
    "average": "average",
    "superior": "superior",
    "genius": "genius"
}

participants = []
for p, s in samples:
    if p not in special_names and p not in participants:
        participants.append(p)

player_map = {
    name: f"player{i+1}"
    for i, name in enumerate(participants)
}

def display_label(participant, soal):
    if participant in special_names:
        return special_names[participant]
    return f"{player_map[participant]}_task{soal}"

# =========================================
# FUNCTION: NEO4J → GRAPH
# =========================================
def get_graph(participant, soal):
    G = nx.DiGraph()

    query = """
    MATCH (a:AOI {participant:$participant, soal:$soal})-[t:TRANSITION]->(b:AOI {participant:$participant, soal:$soal})
    RETURN a.name AS from_name,
           b.name AS to_name,
           t.count AS count
    """

    with driver.session() as session:
        result = session.run(query, participant=participant, soal=soal)

        for record in result:
            a = record["from_name"]
            b = record["to_name"]
            w = record["count"] if record["count"] else 1
            G.add_edge(a, b, weight=w)

    return G

# =========================================
# RELABEL GRAPH
# =========================================
def relabel_graph(G):
    nodes = list(G.nodes())

    # Pastikan "Model" selalu menjadi node pertama jika ada
    nodes_sorted = ["Model"] + [n for n in nodes if n != "Model"] if "Model" in nodes else nodes

    mapping = {node: i for i, node in enumerate(nodes_sorted)}
    G = nx.relabel_nodes(G, mapping)

    # Beri label string untuk Graph2Vec
    nx.set_node_attributes(G, {i: str(i) for i in G.nodes()}, "label")

    return G

# =========================================
# BUILD GRAPH LIST
# =========================================
graphs = []
labels = []

for participant, soal in samples:
    G = get_graph(participant, soal)

    if len(G.nodes()) == 0:
        print(f"⚠️ Graph kosong: {participant} Soal {soal}")
        continue

    G = relabel_graph(G)
    graphs.append(G)
    labels.append(display_label(participant, soal))

if len(graphs) < 2:
    raise ValueError("Graph tidak cukup untuk Graph2Vec dan clustering.")

# =========================================
# GRAPH2VEC
# =========================================
model = Graph2Vec(
    dimensions=64,
    workers=1,      # lebih aman agar tidak error di beberapa environment
    min_count=1,
    epochs=50
)

model.fit(graphs)
embeddings = model.get_embedding()

print("\n=== EMBEDDING MATRIX (64 DIM) ===\n")
for i, label in enumerate(labels):
    print(f"{label} → {embeddings[i]}")

# =========================================
# CLUSTERING
# =========================================
n_clusters = 2 if len(labels) >= 2 else 1

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(embeddings)

print("\n=== CLUSTER RESULT ===\n")
for i in range(len(labels)):
    print(f"{labels[i]} → Cluster {clusters[i]}")

# =========================================
# COSINE SIMILARITY
# =========================================
sim_matrix = cosine_similarity(embeddings)

print("\n=== COSINE SIMILARITY MATRIX ===\n")
for i in range(len(labels)):
    row = ""
    for j in range(len(labels)):
        row += f"{sim_matrix[i][j]:.2f}  "
    print(f"{labels[i]:20s}: {row}")

# =========================================
# MOST SIMILAR PAIRS
# =========================================
print("\n=== MOST SIMILAR PAIRS ===\n")
n = len(labels)

for i in range(n):
    for j in range(i + 1, n):
        print(f"{labels[i]} vs {labels[j]} = {sim_matrix[i][j]:.3f}")

# =========================================
# SIMILARITY TO KEY
# =========================================
if KEY_LABEL in labels:
    key_index = labels.index(KEY_LABEL)

    print(f"\n=== SIMILARITY TO KEY ({KEY_LABEL}) ===\n")
    for i in range(len(labels)):
        print(f"{labels[i]} → {sim_matrix[key_index][i]:.3f}")

# =========================================
# RANKING TO KEY
# =========================================
if KEY_LABEL in labels:
    key_index = labels.index(KEY_LABEL)

    ranking = []
    for i in range(len(labels)):
        if labels[i] == KEY_LABEL:
            continue
        ranking.append((labels[i], sim_matrix[key_index][i]))

    ranking.sort(key=lambda x: x[1], reverse=True)

    print(f"\n=== RANKING SIMILARITY TO KEY ({KEY_LABEL}) ===\n")
    print(f"{'Rank':<5} {'Label':<20} {'Similarity':>10}")
    print("-" * 40)

    for rank, (label, sim) in enumerate(ranking, start=1):
        print(f"{rank:<5} {label:<20} {sim:>10.3f}")

# =========================================
# PCA VISUALIZATION
# =========================================
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(18, 12))

# agar label tidak terlalu menumpuk, beri offset bergantian
offsets = [
    (0, 12), (12, 0), (0, -14), (-14, 0),
    (10, 10), (-10, 10), (10, -10), (-10, -10)
]

for i in range(len(emb_2d)):
    x, y = emb_2d[i]
    dx, dy = offsets[i % len(offsets)]

    if labels[i] == KEY_LABEL:
        ax.scatter(
            x, y,
            marker="*",
            s=1200,
            linewidths=2,
            edgecolors="black",
            zorder=3
        )
        ax.annotate(
            labels[i],
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            fontsize=14,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
        )
    else:
        ax.scatter(
            x, y,
            s=300,
            alpha=0.9,
            zorder=3
        )
        ax.annotate(
            f"{labels[i]} (C{clusters[i]})",
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
        )

ax.set_title("Graph2Vec Embedding (PCA)", fontsize=22, fontweight="bold")
ax.set_xlabel("Principal Component 1", fontsize=16)
ax.set_ylabel("Principal Component 2", fontsize=16)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.show()

# =========================================
# SIMILARITY MATRIX HEATMAP
# =========================================
fig, ax = plt.subplots(figsize=(16, 14))

im = ax.imshow(sim_matrix, aspect="auto", interpolation="nearest")

cbar = fig.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=12)

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90, fontsize=11)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=11)

ax.set_title("Cosine Similarity Matrix", fontsize=22, fontweight="bold")
ax.tick_params(axis='both', labelsize=11)

plt.tight_layout()
plt.show()

# =========================================
# CLOSE
# =========================================
driver.close()