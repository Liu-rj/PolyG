import json
import os
import argparse
import csv


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--path", type=str, default="../datasets/physics", required=True
)
args = argparser.parse_args()

# load the json file
with open(os.path.join(args.path, "graph.json")) as f:
    data = json.load(f)

for key in data.keys():
    print(key, len(data[key].keys()))

# convert this json file to a csv file
label = os.path.basename(args.path).lower()

# add nodes
node_set = set()
node_csv_data = [[f"id:ID({label})", "node_type", "name", "description", ":LABEL"]]
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
            node_csv_data.append(
                [key, node_t, name, description, ";".join([label, node_t])]
            )

csv_nodes = os.path.join(args.path, "nodes.csv")
with open(csv_nodes, "w", newline="", encoding="utf-8") as cf:
    writer = csv.writer(cf)
    writer.writerows(node_csv_data)

# add edges
edge_csv_data = [[f":START_ID({label})", f":END_ID({label})", "relation:TYPE"]]
for node_type in data.keys():
    for key, value in data[node_type].items():
        if key not in node_set:
            continue
        for relation, neighbors in data[node_type][key]["neighbors"].items():
            assert isinstance(neighbors, list)
            for tgt in neighbors:
                if tgt in node_set:
                    edge_csv_data.append([key, tgt, relation])

csv_edges = os.path.join(args.path, "edges.csv")
with open(csv_edges, "w", newline="", encoding="utf-8") as cf:
    writer = csv.writer(cf)
    writer.writerows(edge_csv_data)
