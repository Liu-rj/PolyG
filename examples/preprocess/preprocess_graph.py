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
argparser.add_argument(
    "--path", type=str, default="dataset/maple/Physics", required=True
)
args = argparser.parse_args()

# load the json file
with open(os.path.join(args.path, "graph.json")) as f:
    data = json.load(f)

for key in data.keys():
    print(key, len(data[key].keys()))

# construct a networkx graph using this json file
G = nx.DiGraph()

# add nodes
for node_type in data.keys():
    for key, value in data[node_type].items():
        node_data = data[node_type][key]["features"]
        node_data["node_type"] = node_type.split("_")[0]
        if "maple" in args.path and node_type == "paper_nodes":
            node_data["label"] = ", ".join(node_data["label"])
        if "title" in node_data and "name" not in node_data:
            node_data["name"] = node_data["title"]
            del node_data["title"]
        if "abstract" in node_data and "description" not in node_data:
            node_data["description"] = node_data["abstract"]
            del node_data["abstract"]
        G.add_node(key, **node_data)

print("# nodes:", G.number_of_nodes())
print("# edges:", G.number_of_edges())

# add edges
for node_type in data.keys():
    for key, value in data[node_type].items():
        for relation, neighbors in data[node_type][key]["neighbors"].items():
            if isinstance(neighbors, list):
                for edge in neighbors:
                    if edge in G.nodes:
                        G.add_edge(key, edge, relation=relation)
            else:
                if edge in G.nodes:
                    G.add_edge(key, edge, relation=relation)

print("# nodes:", G.number_of_nodes())
print("# edges:", G.number_of_edges())
print(f"graph is directed: {G.is_directed()}")

del data

pickle.dump(G, open(os.path.join(args.path, "graph.pkl"), "wb"))

exit()


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
    "relation": [G.edges.get((e[0], e[1])).get("relation", "UNKOWN") for e in all_edges]
}
del G

# all_nodes_data = [{"id": nid, **data} for nid, data in zip(all_nodes, all_nodes_data)]
keys = ["name", "node_type", "description"]
ig_nodes_data = {k: [d.get(k, "UNKOWN") for d in all_nodes_data] for k in keys}
del all_nodes_data
ig_nodes_data["node_name"] = ig_nodes_data.pop("name")

# add node and edge list
G_ig.add_vertices(all_nodes, attributes=ig_nodes_data)
print(G_ig.summary())
G_ig.add_edges(all_edges, all_edges_data)

e2r = get_entities_to_relationships_map(G_ig)

# # Print summary
print(G_ig.summary())
ig.Graph.write_picklez(G_ig, os.path.join(args.path, "graph_igraph_data.pklz"))  # type: ignore

with open(os.path.join(args.path, "map_e2r_blob_data.pkl"), "wb") as f:
    pickle.dump(e2r, f)
