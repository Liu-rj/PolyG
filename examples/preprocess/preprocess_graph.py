import json
import networkx as nx
import pickle
import os
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument("--path", type=str, default="physics", required=True)
args = argparser.parse_args()

# load the json file
with open(os.path.join(args.path, "graph.json")) as f:
    data = json.load(f)

for key in data.keys():
    print(key, len(data[key].keys()))

# construct a networkx graph using this json file
G = nx.Graph()

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

pickle.dump(G, open(os.path.join(args.path, "graph.pkl"), "wb"))
