import random
import jsonlines
import argparse
import pickle
import os
import networkx as nx
from typing import List
from neo4j import GraphDatabase


neo4j_config = {
    "neo4j_url": os.environ.get("NEO4J_URL", "neo4j://localhost:7687"),
    "neo4j_auth": (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "123456789"),
    ),
}


single_entity_concrete_template = {
    "physics": {
        "author": {
            "What paper have the author '{}' published?": {
                "cypher": """
            MATCH (author:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author {{id: '{}'}})
            -[:paper]-(paper:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper)
            RETURN paper.name as name
            """,
                "hops": 1,
            },
            "What are the academic collaborators of '{}'?": {
                "cypher": """
            MATCH (author:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author {{id: '{}'}})
            -[:paper]-(paper:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper)
            -[:paper]-(collaborator:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author)
            WHERE collaborator <> author
            RETURN DISTINCT collaborator.name AS name
            """,
                "hops": 2,
            },
            "What venues have the author '{}' published in?": {
                "cypher": """
            MATCH (author:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author {{id: '{}'}})
            -[:paper]-(paper:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper)
            -[:paper]-(venue:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:venue)
            RETURN DISTINCT venue.name AS name
            """,
                "hops": 2,
            },
        },
        "paper": {
            "Who are the authors of the paper '{}'?": {
                "cypher": """
            MATCH (p:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper {{id: '{}'}})
            -[:paper]-(a:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author)
            RETURN a.name AS name
            """,
                "hops": 1,
            },
            "Where is the paper '{}' published?": {
                "cypher": """
            MATCH (p:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper {{id: '{}'}})
            -[:paper]-(v:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:venue)
            RETURN v.name AS name
            """,
                "hops": 1,
            },
            "Who are the academic collaborators of the author who writes the paper '{}'?": {
                "cypher": """
            MATCH (p:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper {{id: '{}'}})
            MATCH (p)-[:paper]-(a:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author)
            MATCH (a)-[:paper]-(otherPaper:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper)
            MATCH (otherPaper)-[:paper]-(coAuthor:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author)
            WHERE coAuthor <> a
            RETURN DISTINCT coAuthor.name AS name
            """,
                "hops": 3,
            },
            "What venues have the author of the paper '{}' published in?": {
                "cypher": """
            MATCH (p:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper {{id: '{}'}})
            -[:paper]-(a:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author)
            -[:paper]-(other_p:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper)
            -[:paper]-(v:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:venue)
            RETURN DISTINCT v.name AS name
            """,
                "hops": 3,
            },
            "What venues have the academic collaborators of the author who writes the paper '{}' published in?": {
                "cypher": """
            MATCH (start_paper:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper {{id: '{}'}})
            -[:paper]-(author:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author)
            -[:paper]-(collab_paper:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper)
            -[:paper]-(collaborator:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author)
            WHERE collaborator <> author
            MATCH (collaborator)-[:paper]-(pub:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper)
            -[:paper]-(venue:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:venue)
            RETURN DISTINCT venue.name AS name
            """,
                "hops": 5,
            },
        },
    }
}


multi_entity_concrete_template = {
    "physics": {
        "What is the relationship between '{}' and '{}' regarding collaborated papers?": {
            "cypher": """
            MATCH (author1:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author)
            -[:paper]-(paper1:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper)
            -[:paper]-(author2:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author)
            WHERE author1 <> author2
            RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
            """,
            "hops": 2,
        },
        "What is the relationship between '{}' and '{}' regarding paper references?": {
            "cypher": """
            MATCH (author1:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author)
            -[:paper]-(paper1:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper)
            -[:reference|cited_by]-(paper2:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper)
            -[:paper]-(author2:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author)
            WHERE author1 <> author2
            RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
            """,
            "hops": 3,
        },
        "What is the relationship between '{}' and '{}' regarding common collaborators?": {
            "cypher": """
            MATCH (author1:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author)
            -[:paper]-(paper1:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper)
            -[:paper]-(collaborator:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author)
            -[:paper]-(paper2:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper)
            -[:paper]-(author2:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author)
            WHERE author1 <> collaborator AND author2 <> collaborator
            RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
            """,
            "hops": 4,
        },
        "What is the relationship between '{}' and '{}' regarding common venues they have published in?": {
            "cypher": """
            MATCH (author1:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author)
            -[:paper]-(paper1:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper)
            -[:paper]-(venue:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:venue)
            -[:paper]-(paper2:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:paper)
            -[:paper]-(author2:___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:author)
            WHERE author1 <> author2
            RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
            """,
            "hops": 4,
        },
    }
}


def gen_single_entity_abstract(graph: nx.Graph, n: int, output_path: str):
    all_nodes = list(graph.nodes())
    questions = []
    for i in range(n):
        node = random.choice(all_nodes)
        node_data = graph.nodes[node]
        question = f"Tell me about '{node_data['name']}'."
        questions.append(
            {
                "qid": i,
                "question": question,
                "entity": {node_data["name"]: node},
                "type": "single_entity_abstract",
                "hops": 1,
                "answer": "N/A",
            }
        )
    with jsonlines.open(output_path, "w") as writer:
        for row in questions:
            writer.write(row)


def gen_single_entity_concrete(
    graph: nx.Graph, n: int, graph_name: str, output_path: str
):
    neo4j_url = neo4j_config["neo4j_url"]
    neo4j_auth = neo4j_config["neo4j_auth"]
    driver = GraphDatabase.driver(
        neo4j_url,
        auth=neo4j_auth,
        max_connection_pool_size=100,
        connection_timeout=60,
    )
    q_templates = single_entity_concrete_template[graph_name]
    questions = []
    for node_type, templates in q_templates.items():
        for q, content in templates.items():
            q_cypher, n_hop = content["cypher"], content["hops"]
            i = 0
            while i < n:
                node = random.choice(list(graph.nodes()))
                while graph.nodes[node]["node_type"] != node_type:
                    node = random.choice(list(graph.nodes()))

                result_names = []
                continue_flag = False
                with driver.session() as session:
                    try:
                        with session.begin_transaction(timeout=30) as tx:
                            result = tx.run(q_cypher.format(node))
                            for record in result:
                                result_names.append(record["name"])
                                if len(result_names) > 20:
                                    print(f"{i}: Too many results, retry")
                                    continue_flag = True
                                    break
                    except Exception as e:
                        print(f"Query failed: {e}")
                        continue_flag = True
                if continue_flag or len(result_names) == 0:
                    continue

                node_name = graph.nodes[node]["name"]
                question = q.format(node_name)
                q_entity = {
                    "qid": len(questions),
                    "question": question,
                    "entity": {node_name: node},
                    "type": "single_entity_concrete",
                    "hops": n_hop,
                    "answer": ", ".join(result_names),
                }
                questions.append(q_entity)
                print(q_entity)
                i += 1

    with jsonlines.open(output_path, "w") as writer:
        for row in questions:
            writer.write(row)


def gen_multi_entity_abstract(
    graph: nx.Graph, n: int, hops: List[int], output_path: str
):
    def bfs_with_path_length(graph, start_node, length):
        queue = [(start_node, 0)]
        visited = {graph.nodes.get(start_node)["name"]}

        while queue:
            current_node, current_length = queue.pop(0)
            if current_length == length:
                return current_node
            elif current_length < length:
                for neighbor in graph.neighbors(current_node):
                    name = graph.nodes[neighbor]["name"]
                    if name not in visited:
                        visited.add(name)
                        queue.append((neighbor, current_length + 1))
        return None

    all_nodes = list(graph.nodes())
    questions = []
    for it, n_hop in enumerate(hops):
        i = 0
        while i < n:
            start_node = random.choice(all_nodes)
            end_node = bfs_with_path_length(graph, start_node, n_hop)
            if end_node is None:
                continue
            start_node_data = graph.nodes[start_node]
            end_node_data = graph.nodes[end_node]
            question = f"What is the relationship between '{start_node_data['name']}' and '{end_node_data['name']}'?"
            q_entity = {
                "qid": i + it * n,
                "question": question,
                "entity": {
                    start_node_data["name"]: start_node,
                    end_node_data["name"]: end_node,
                },
                "type": "multi_entity_abstract",
                "hops": n_hop,
                "answer": "N/A",
            }
            questions.append(q_entity)
            print(q_entity)
            i += 1

    with jsonlines.open(output_path, "w") as writer:
        for row in questions:
            writer.write(row)


def gen_multi_entity_concrete(
    graph: nx.Graph, n: int, graph_name: str, output_path: str
):
    neo4j_url = neo4j_config["neo4j_url"]
    neo4j_auth = neo4j_config["neo4j_auth"]
    driver = GraphDatabase.driver(
        neo4j_url,
        auth=neo4j_auth,
        max_connection_pool_size=100,
        connection_timeout=60,
    )
    q_templates = multi_entity_concrete_template[graph_name]
    questions = []
    for q, content in q_templates.items():
        q_cypher, n_hop = content["cypher"], content["hops"]

        result_list = []
        name_set = set()
        with driver.session() as session:
            results = session.run(q_cypher)
            for record in results:
                if record["name1"] in name_set or record["name2"] in name_set:
                    continue
                name_set.add(record["name1"])
                name_set.add(record["name2"])
                result_list.append(record)
                if len(result_list) == 20:
                    break

        for line in result_list:
            question = q.format(line["name1"], line["name2"])
            q_entity = {
                "qid": len(questions),
                "question": question,
                "entity": {
                    line["name1"]: line["id1"],
                    line["name2"]: line["id2"],
                },
                "type": "single_entity_concrete",
                "hops": n_hop,
                "answer": "N/A",
            }
            questions.append(q_entity)
            print(q_entity)

    with jsonlines.open(output_path, "w") as writer:
        for row in questions:
            writer.write(row)


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--path", type=str, default="datasets/maple/Physics", required=True
)
args = argparser.parse_args()
print(args)

# load the graph
graph = pickle.load(open(os.path.join(args.path, "graph.pkl"), "rb"))
print("NetworkX graph loaded")
print("# nodes:", graph.number_of_nodes())
print("# edges:", graph.number_of_edges())

# generate questions
# gen_single_entity_abstract(
#     graph, 100, os.path.join(args.path, "single_entity_abstract.jsonl")
# )
# gen_single_entity_concrete(
#     graph, 10, "physics", os.path.join(args.path, "single_entity_concrete.jsonl")
# )
# gen_multi_entity_abstract(
#     graph, 20, [2, 3, 4, 5], os.path.join(args.path, "multi_entity_abstract.jsonl")
# )
gen_multi_entity_concrete(
    graph, 10, "physics", os.path.join(args.path, "multi_entity_concrete.jsonl")
)
