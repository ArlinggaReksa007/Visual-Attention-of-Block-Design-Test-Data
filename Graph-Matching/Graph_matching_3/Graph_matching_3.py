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
# CONFIG
# =========================================
URI = "bolt://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "29051999"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

AOI_LIST = ["Area 1", "Area 2", "Area 3", "Area 4", "Model"]

samples = [
    ("Candra",1),("Candra",2),("Candra",3),("Candra",4),
    ("Arif",1),("Arif",2),("Arif",3),("Arif",4),
    ("Amar",1),("Amar",2),("Amar",3),("Amar",4),
    ("Bayu",1),("Bayu",2),("Bayu",3),("Bayu",4),
    ("genius",1),("superior",1),("average",1),
]

KEY = ("Kunci",1)
ALPHA = 0.5


# =========================================
# BUILD GRAPH
# =========================================
def build_graph(p, s):

    AOI_EXT = [a+"_out" for a in AOI_LIST] + [a+"_in" for a in AOI_LIST]

    features = {a:[0,0] for a in AOI_LIST}
    transitions = {}

    with driver.session() as session:

        q1 = """
        MATCH (:Participant {name:$p, soal:$s})-[r:LOOKED_AT]->(a:AOI {participant:$p, soal:$s})
        RETURN a.name AS name, r.total_time AS t, r.fixation_count AS f
        """
        for r in session.run(q1, p=p, s=s):
            if r["name"] in features:
                features[r["name"]] = [float(r["t"] or 0), float(r["f"] or 0)]

        q2 = """
        MATCH (a:AOI {participant:$p, soal:$s})-[t:TRANSITION]->(b:AOI {participant:$p, soal:$s})
        RETURN a.name AS a, b.name AS b, t.count AS c
        """
        for r in session.run(q2, p=p, s=s):
            transitions[(r["a"]+"_out", r["b"]+"_in")] = float(r["c"] or 0)

    X = []
    for a in AOI_LIST:
        X.append(features[a])
        X.append(features[a])
    X = np.array(X)

    G = nx.Graph()
    G.add_nodes_from(AOI_EXT)

    for (u,v),c in transitions.items():
        G.add_edge(u,v,weight=1/(c+1))

    n = len(AOI_EXT)
    C = np.full((n,n), np.inf)

    for i,src in enumerate(AOI_EXT):
        dist = nx.single_source_dijkstra_path_length(G,src,weight="weight")
        for j,dst in enumerate(AOI_EXT):
            if dst in dist:
                C[i,j] = dist[dst]

    C[np.isinf(C)] = np.nanmax(C[np.isfinite(C)])*1.5
    np.fill_diagonal(C,0)
    C = C/(C.max()+1e-9)

    return C,X


# =========================================
# FGW
# =========================================
def fgw(C1,X1,C2,X2):
    p = np.ones(len(X1))/len(X1)
    q = np.ones(len(X2))/len(X2)
    M = ot.dist(X1,X2)
    return ot.gromov.fused_gromov_wasserstein2(M,C1,C2,p,q,alpha=ALPHA)


# =========================================
# LOAD DATA
# =========================================
items = samples + [KEY]

graphs = []
labels = []

for p,s in items:
    C,X = build_graph(p,s)
    graphs.append((C,X))
    labels.append("Kunci" if (p,s)==KEY else f"{p}_S{s}")


# =========================================
# NORMALIZE
# =========================================
allX = np.vstack([g[1] for g in graphs])
scaler = StandardScaler().fit(allX)

graphs = [(C,scaler.transform(X)) for C,X in graphs]


# =========================================
# DISTANCE MATRIX
# =========================================
n = len(graphs)
dist = np.zeros((n,n))

for i in range(n):
    for j in range(i,n):
        d = fgw(graphs[i][0],graphs[i][1],
                graphs[j][0],graphs[j][1])
        dist[i,j]=dist[j,i]=d

sim = 1/(1+dist)
key_idx = labels.index("Kunci")


# =========================================
# PRINT MATRIX
# =========================================
print("\n=== SIMILARITY MATRIX (FGW) ===\n")
plt.figure(figsize=(10, 8))
plt.imshow(sim, cmap="magma", aspect="auto")
plt.colorbar(label="Similarity")

plt.xticks(range(n), labels, rotation=45, ha="right")
plt.yticks(range(n), labels)

for i in range(n):
    for j in range(n):
        val = sim[i, j]
        text_color = "white" if val < 0.5 else "black"
        plt.text(j, i, f"{val:.2f}", 
                 ha="center", va="center", 
                 color=text_color, fontsize=8)

plt.title("Similarity Antar Peserta (FGW)")
plt.tight_layout()
plt.show()
# =========================================
# RANKING
# =========================================
print("\n=== RANKING KE KUNCI ===")

ranking = []
for i,l in enumerate(labels):
    if l=="Kunci": continue
    s = sim[i,key_idx]
    ranking.append((l,s))

ranking.sort(key=lambda x:x[1],reverse=True)

for l,s in ranking:
    print(f"{l:12s} -> {s:.3f}")


# =========================================
# NEAREST NEIGHBOR
# =========================================
print("\n=== NEAREST NEIGHBOR ===")

topk_mean = []

for i in range(n):
    sims = sim[i].copy()
    sims[i]=0

    idx = np.argsort(sims)[-3:]
    mean = sims[idx].mean()
    topk_mean.append(mean)

    neigh = [labels[j] for j in idx]
    print(f"{labels[i]:12s} -> {neigh}")


# =========================================
# ANCHOR PLOT
# =========================================
plt.figure(figsize=(10,7))

for i,l in enumerate(labels):

    if l=="Kunci":
        x,y = 1,1
        plt.scatter(x,y,s=300,marker="*",color="red")
        plt.text(x,y,"Kunci",ha="center")
    else:
        x = sim[i,key_idx]
        y = topk_mean[i]

        plt.scatter(x,y,s=80)
        plt.text(x,y,l,fontsize=8)

plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("Similarity ke Kunci")
plt.ylabel("Kedekatan dengan Graph Lain")
plt.title("Anchor Plot FGW")

plt.grid(alpha=0.3)
plt.show()

driver.close()