# =========================================
# IMPORT
# =========================================
from neo4j import GraphDatabase
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ot
from sklearn.preprocessing import StandardScaler

# =========================================
# PLOT STYLE
# =========================================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

# =========================================
# CONFIG
# =========================================
URI = "bolt://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "29051999"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# optional: cek koneksi lebih awal
try:
    driver.verify_connectivity()
    print("✓ Connected to Neo4j")
except Exception as e:
    print("✗ Failed to connect to Neo4j")
    print(e)
    raise SystemExit

AOI_LIST = ["Area 1", "Area 2", "Area 3", "Area 4", "Model"]

samples = [
    ("Candra", 1), ("Candra", 2), ("Candra", 3), ("Candra", 4),
    ("Arif", 1), ("Arif", 2), ("Arif", 3), ("Arif", 4),
    ("Amar", 1), ("Amar", 2), ("Amar", 3), ("Amar", 4),
    ("Bayu", 1), ("Bayu", 2), ("Bayu", 3), ("Bayu", 4),
    ("genius", 1),
    ("superior", 1),
    ("average", 1),
]

KEY = ("Kunci", 1)
KEY_LABEL = "key"

ALPHA = 0.5

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
# BUILD GRAPH
# =========================================
def build_graph(p, s):
    AOI_EXT = (
        [a + "_out" for a in AOI_LIST] +
        [a + "_in" for a in AOI_LIST]
    )

    features = {a: [0, 0] for a in AOI_LIST}
    transitions = {}

    with driver.session() as session:
        q1 = """
        MATCH (:Participant {name:$p, soal:$s})
              -[r:LOOKED_AT]->
              (a:AOI {participant:$p, soal:$s})
        RETURN
            a.name AS name,
            r.total_time AS t,
            r.fixation_count AS f
        """

        for r in session.run(q1, p=p, s=s):
            if r["name"] in features:
                features[r["name"]] = [
                    float(r["t"] or 0),
                    float(r["f"] or 0)
                ]

        q2 = """
        MATCH (a:AOI {participant:$p, soal:$s})
              -[t:TRANSITION]->
              (b:AOI {participant:$p, soal:$s})
        RETURN
            a.name AS a,
            b.name AS b,
            t.count AS c
        """

        for r in session.run(q2, p=p, s=s):
            transitions[(r["a"] + "_out", r["b"] + "_in")] = float(r["c"] or 0)

    X = []
    for a in AOI_LIST:
        X.append(features[a])
        X.append(features[a])
    X = np.array(X, dtype=float)

    G = nx.Graph()
    G.add_nodes_from(AOI_EXT)

    for (u, v), c in transitions.items():
        G.add_edge(u, v, weight=1 / (c + 1))

    n = len(AOI_EXT)
    C = np.full((n, n), np.inf)

    for i, src in enumerate(AOI_EXT):
        dist = nx.single_source_dijkstra_path_length(G, src, weight="weight")
        for j, dst in enumerate(AOI_EXT):
            if dst in dist:
                C[i, j] = dist[dst]

    finite_vals = C[np.isfinite(C)]
    if len(finite_vals) > 0:
        C[np.isinf(C)] = np.max(finite_vals) * 1.5

    np.fill_diagonal(C, 0)
    C = C / (C.max() + 1e-9)

    return C, X

# =========================================
# FGW DISTANCE
# =========================================
def fgw(C1, X1, C2, X2):
    p = np.ones(len(X1)) / len(X1)
    q = np.ones(len(X2)) / len(X2)

    M = ot.dist(X1, X2)

    return ot.gromov.fused_gromov_wasserstein2(
        M,
        C1,
        C2,
        p,
        q,
        alpha=ALPHA
    )

# =========================================
# LOAD DATA
# =========================================
items = samples + [KEY]

graphs = []
labels = []

for p, s in items:
    C, X = build_graph(p, s)
    graphs.append((C, X))
    labels.append(KEY_LABEL if (p, s) == KEY else display_label(p, s))

# =========================================
# NORMALIZE FEATURE
# =========================================
allX = np.vstack([g[1] for g in graphs])
scaler = StandardScaler().fit(allX)

graphs = [(C, scaler.transform(X)) for C, X in graphs]

# =========================================
# DISTANCE MATRIX
# =========================================
n = len(graphs)
dist = np.zeros((n, n))

for i in range(n):
    for j in range(i, n):
        d = fgw(graphs[i][0], graphs[i][1], graphs[j][0], graphs[j][1])
        dist[i, j] = d
        dist[j, i] = d

sim = 1 / (1 + dist)
key_idx = labels.index(KEY_LABEL)

# =========================================
# SIMILARITY MATRIX
# =========================================
print("\n=== SIMILARITY MATRIX (FGW) ===\n")

fig, ax = plt.subplots(figsize=(16, 13))
im = ax.imshow(sim, cmap="magma", aspect="auto", interpolation="nearest")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Similarity", fontsize=14)
cbar.ax.tick_params(labelsize=12)

ax.set_xticks(range(n))
ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=10)

ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=10)

ax.set_title("FGW Similarity Matrix", fontsize=22, fontweight="bold")

for i in range(n):
    for j in range(n):
        val = sim[i, j]
        text_color = "white" if val < 0.5 else "black"
        ax.text(
            j, i, f"{val:.2f}",
            ha="center", va="center",
            color=text_color,
            fontsize=9,
            fontweight="bold"
        )

plt.tight_layout()
plt.show()

# =========================================
# RANKING TO KEY
# =========================================
print("\n=== RANKING TO KEY ===\n")

ranking = []
for i, label in enumerate(labels):
    if label == KEY_LABEL:
        continue
    ranking.append((label, sim[i, key_idx]))

ranking.sort(key=lambda x: x[1], reverse=True)

for rank, (label, score) in enumerate(ranking, start=1):
    print(f"{rank:2d}. {label:20s} {score:.3f}")

# =========================================
# NEAREST NEIGHBOR
# =========================================
print("\n=== NEAREST NEIGHBOR ===\n")

topk_mean = []

for i in range(n):
    sims = sim[i].copy()
    sims[i] = 0

    idx = np.argsort(sims)[-3:]
    mean_val = sims[idx].mean()
    topk_mean.append(mean_val)

    neighbors = [labels[j] for j in idx]
    print(f"{labels[i]:20s} -> {neighbors}")

# =========================================
# ANCHOR PLOT
# =========================================
fig, ax = plt.subplots(figsize=(14, 10))

offsets = [
    (0, 12), (12, 0), (0, -14), (-14, 0),
    (10, 10), (-10, 10), (10, -10), (-10, -10)
]

for i, label in enumerate(labels):
    if label == KEY_LABEL:
        ax.scatter(
            1, 1,
            s=1200,
            marker="*",
            color="red",
            edgecolors="black",
            linewidths=1.5,
            zorder=3
        )
        ax.annotate(
            KEY_LABEL,
            (1, 1),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
            fontsize=14,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
        )
    else:
        x = sim[i, key_idx]
        y = topk_mean[i]
        dx, dy = offsets[i % len(offsets)]

        ax.scatter(
            x, y,
            s=180,
            alpha=0.9,
            zorder=3
        )
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
        )

ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Similarity to Key", fontsize=16)
ax.set_ylabel("Similarity to Other Graphs", fontsize=16)
ax.set_title("FGW Anchor Plot", fontsize=22, fontweight="bold")
ax.grid(alpha=0.3)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.show()

# =========================================
# CLOSE
# =========================================
driver.close()