import json
import networkx as nx
import igraph as ig
import numpy as np
import pickle
import os
import argparse
from typing import List, Tuple
from scipy.sparse import csr_matrix


argparser = argparse.ArgumentParser()
argparser.add_argument("--path", type=str, default="../datasets/physics", required=True)
args = argparser.parse_args()

# load the json file
with open(os.path.join(args.path, "graph.json")) as f:
    data = json.load(f)

for key in data.keys():
    print(key, len(data[key].keys()))

# construct a networkx graph using this json file
G = nx.DiGraph()

# add nodes
node_set = set()
for node_type in data.keys():
    node_t = node_type.split("_")[0]
    for key, value in data[node_type].items():
        node_data = data[node_type][key]["features"]
        name = node_data["name"] if "name" in node_data else node_data["title"]
        name = name.replace("\n", " ").replace("\r", " ").replace('"', "'")
        if name == "":
            print(f"Empty name for {key}, skipping")
            continue
        description = (
            node_data["description"]
            if "description" in node_data
            else node_data.get("abstract", "UNKNOWN")
        )
        description = (
            description.replace("\n", " ").replace("\r", " ").replace('"', "'")
        )
        if key not in node_set:
            node_set.add(key)
            G.add_node(
                key, **{"name": name, "node_type": node_t, "description": description}
            )

print("# nodes:", G.number_of_nodes())
print("# edges:", G.number_of_edges())

# add edges
for node_type in data.keys():
    for key, value in data[node_type].items():
        if key not in node_set:
            continue
        for relation, neighbors in data[node_type][key]["neighbors"].items():
            assert isinstance(neighbors, list)
            for tgt in neighbors:
                if tgt in node_set:
                    G.add_edge(key, tgt, relation=relation)

print("# nodes:", G.number_of_nodes())
print("# edges:", G.number_of_edges())
print(f"graph is directed: {G.is_directed()}")

del data

pickle.dump(G, open(os.path.join(args.path, "graph.pkl"), "wb"))


def csr_from_indices_list(data: List[List[int]], shape: Tuple[int, int]) -> csr_matrix:
    """Create a CSR matrix from a list of lists."""
    num_rows = len(data)

    # Flatten the list of lists and create corresponding row indices
    row_indices = np.repeat(np.arange(num_rows), [len(row) for row in data])
    col_indices = np.concatenate(data) if num_rows > 0 else np.array([], dtype=np.int64)

    # Data values (all ones in this case)
    values = np.broadcast_to(1, len(row_indices))

    # Create the CSR matrix
    return csr_matrix((values, (row_indices, col_indices)), shape=shape)


def get_entities_to_relationships_map(graph: ig.Graph) -> csr_matrix:
    if len(graph.vs) == 0:  # type: ignore
        return csr_matrix((0, 0))

    return csr_from_indices_list(
        [
            [edge.index for edge in vertex.incident()]  # type: ignore
            for vertex in graph.vs  # type: ignore
        ],
        shape=(graph.vcount(), graph.ecount()),  # type: ignore
    )


G_ig = ig.Graph(directed=G.is_directed())

all_nodes = list(G.nodes())
all_nodes_data = [G.nodes.get(nid) for nid in all_nodes]
all_edges = list(G.edges())
all_edges_data = {
    "relation": [G.edges.get((e[0], e[1])).get("relation", "UNKOWN") for e in all_edges]  # type: ignore
}
del G

keys = ["name", "node_type", "description"]
ig_nodes_data = {k: [d.get(k, "UNKOWN") for d in all_nodes_data] for k in keys}  # type: ignore
ig_nodes_data["node_name"] = ig_nodes_data.pop("name")

# add node and edge list
G_ig.add_vertices(all_nodes, attributes=ig_nodes_data)
print(G_ig.summary())
G_ig.add_edges(all_edges, all_edges_data)

del all_nodes, all_edges, all_nodes_data, all_edges_data

e2r = get_entities_to_relationships_map(G_ig)

# # Print summary
print(G_ig.summary())
ig.Graph.write_picklez(G_ig, os.path.join(args.path, "graph_igraph_data.pklz"))  # type: ignore

with open(os.path.join(args.path, "map_e2r_blob_data.pkl"), "wb") as f:
    pickle.dump(e2r, f)
