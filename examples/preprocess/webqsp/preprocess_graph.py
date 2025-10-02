import json
import networkx as nx
import igraph as ig
import numpy as np
import pickle
import os
import argparse
from datasets import load_dataset
from typing import List, Tuple
from scipy.sparse import csr_matrix


argparser = argparse.ArgumentParser()
argparser.add_argument("--benchmark", type=str, default="webqsp", required=True)
argparser.add_argument("--path", type=str, default="dataset/webqsp", required=True)
args = argparser.parse_args()


def build_graph(graph: list) -> nx.DiGraph:
    G = nx.DiGraph()
    for triplet in graph:
        h, r, t = triplet
        characters_to_replace = [".", "-", "#", " "]
        for char in characters_to_replace:
            r = r.replace(char, "_")
        G.add_edge(h, t, relation=r.strip())
        G.add_node(h, **{"name": h, "node_type": "UNKOWN", "description": "UNKOWN"})
        G.add_node(t, **{"name": t, "node_type": "UNKOWN", "description": "UNKOWN"})
    return G


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


dataset = load_dataset(f"rmanluo/RoG-{args.benchmark}", split="test")

# dataset = load_dataset(f"../../webqsp_cwq/datasets/cwq", split="test")

# # load id file
# with open("../../datasets/cwq/test_ids.txt", "r") as f:
#     test_ids = f.read().splitlines()
# print(f"Number of unique test examples: {len(set(test_ids))}")

for it, sample in enumerate(dataset):
    # if sample["id"] not in test_ids:
    #     continue
    # print(f"Processing sample {it} with id {sample['id']}")
    question = sample["question"]
    G = build_graph(sample["graph"])

    print("# nodes:", G.number_of_nodes())
    print("# edges:", G.number_of_edges())
    print(f"graph is directed: {G.is_directed()}")

    nodes = {}
    for node_id, properties in G.nodes(data=True):
        nodes[node_id] = {"features": {"name": properties["name"]}, "neighbors": {}}
        edges = G.edges(node_id, data=True)
        for u, v, edge_data in edges:
            relation = edge_data.get("relation", "UNKOWN")
            if relation not in nodes[node_id]["neighbors"]:
                nodes[node_id]["neighbors"][relation] = []
            nodes[node_id]["neighbors"][relation].append(v)

    print(f"Number of nodes: {len(nodes)}")

    # save the graph in JSON format
    json.dump(
        {
            "nodes": nodes,
        },
        open(os.path.join(args.path, f"graph_{it}.json"), "w"),
        indent=4,
    )

    pickle.dump(G, open(os.path.join(args.path, f"graph_{it}.pkl"), "wb"))

    G_ig = ig.Graph(directed=G.is_directed())

    all_nodes = list(G.nodes())
    all_nodes_data = [G.nodes.get(nid) for nid in all_nodes]
    all_edges = list(G.edges())
    all_edges_data = {
        "relation": [
            G.edges.get((e[0], e[1])).get("relation", "UNKOWN") for e in all_edges  # type: ignore
        ]
    }

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
    ig.Graph.write_picklez(G_ig, os.path.join(args.path, f"graph_igraph_data_{it}.pklz"))  # type: ignore

    with open(os.path.join(args.path, f"map_e2r_blob_data_{it}.pkl"), "wb") as f:
        pickle.dump(e2r, f)
