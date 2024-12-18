import networkx as nx
import igraph as ig
import pickle
from tqdm import tqdm
import os
from typing import List, Tuple, Union
import numpy as np
from scipy.sparse import csr_matrix


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


DATASET_DIR = "datasets/maple/Physics"

# Load the graph
G_nx: nx.Graph = pickle.load(open(f"{DATASET_DIR}/graph.pkl", "rb"))

G_ig = ig.Graph(directed=G_nx.is_directed())

all_nodes = list(G_nx.nodes())
all_nodes_data = [G_nx.nodes.get(nid) for nid in all_nodes]
# all_nodes_data = [{"id": nid, **data} for nid, data in zip(all_nodes, all_nodes_data)]
keys = ["name", "node_type"]
all_nodes_data = {k: [d.get(k, "UNKOWN") for d in all_nodes_data] for k in keys}
all_nodes_data["node_name"] = all_nodes_data.pop("name")

all_edges = list(G_nx.edges())
all_edges_data = {
    "relation": [
        G_nx.edges.get((e[0], e[1])).get("relation", "UNKOWN") for e in all_edges
    ]
}

# add node and edge list
G_ig.add_vertices(all_nodes, attributes=all_nodes_data)
print(G_ig.summary())
G_ig.add_edges(all_edges, all_edges_data)

e2r = get_entities_to_relationships_map(G_ig)

# # Print summary
print(G_ig.summary())
ig.Graph.write_picklez(G_ig, f"{DATASET_DIR}/graph_igraph_data.pklz")  # type: ignore

with open(f"{DATASET_DIR}/map_e2r_blob_data.pkl", "wb") as f:
    pickle.dump(e2r, f)
