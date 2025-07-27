import os
from datasets import load_dataset
import networkx as nx
from neo4j import GraphDatabase
from tqdm import tqdm

neo4j_config = {
    "neo4j_url": os.environ.get("NEO4J_URL", "neo4j://localhost:7687"),
    "neo4j_auth": (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "12345678"),
    ),
}

driver = GraphDatabase.driver(
    neo4j_config["neo4j_url"], auth=neo4j_config["neo4j_auth"]
)


def build_graph(graph: list) -> nx.DiGraph:
    G = nx.DiGraph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G


# Load dataset
dataset = load_dataset("rmanluo/RoG-webqsp", split="test")


def prepare_dataset(sample):
    # Find ground-truth paths for each Q-P pair
    graph = build_graph(sample["graph"])
    return graph


# dataset = dataset.map(
#     prepare_dataset,
#     num_proc=1,
# )

G_networkx = prepare_dataset(dataset[0])
# print(G_networkx.nodes(data=True))
# print(G_networkx.edges(data=True))
# exit()

# all_relation_types = set()
# for u, v, properties in tqdm(G_networkx.edges(data=True), ncols=100):
#     rel_type = properties["relation"]  # Default if not in properties
#     characters_to_replace = [".", "-", "#", " "]
#     for char in characters_to_replace:
#         rel_type = rel_type.replace(char, "_")
#     all_relation_types.add(rel_type)

# print("Number of relation types:", len(all_relation_types))
# for i, rel_type in enumerate(all_relation_types):
#     print(f"{i + 1}. {rel_type}")
# exit()

with driver.session() as session:
    # 1. Create Nodes
    for node_id, properties in tqdm(G_networkx.nodes(data=True), ncols=100):
        node_prop = {"name": node_id}
        # Ensure a label is set for the node
        query = f"MERGE (n:webqsp:node {{id: $node_id}})" "SET n += $properties"
        session.run(query, node_id=node_id, properties=node_prop)

    # 2. Create Relationships
    for u, v, properties in tqdm(G_networkx.edges(data=True), ncols=100):
        # Ensure a relation is set
        rel_type = properties["relation"]  # Default if not in properties
        # rel_type = relation.split(".")[-1]  # Use the last part of the relation
        characters_to_replace = [".", "-", "#", " "]
        for char in characters_to_replace:
            rel_type = rel_type.replace(char, "_")
        # rel_type = relation.replace(".", "_")
        # rel_type = rel_type.replace("-", "_").replace("#", "_").replace(" ", "_")

        query = (
            f"MATCH (a:webqsp:node {{id: $source_id}}), (b:webqsp:node {{id: $target_id}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "  # Using MERGE to avoid duplicate relationships
            # f"SET r += $attr"
        )
        session.run(query, source_id=u, target_id=v)

print("NetworkX graph successfully inserted into Neo4j.")
driver.close()
