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
            MATCH (author:Physics:author {{id: '{}'}})
            -[:paper]->(paper:Physics:paper)
            RETURN paper.name as name
            """,
                "hops": 1,
            },
            "What are the academic collaborators of '{}'?": {
                "cypher": """
            MATCH (author:Physics:author {{id: '{}'}})
            -[:paper]->(paper:Physics:paper)
            -[:author]->(collaborator:Physics:author)
            WHERE collaborator <> author
            RETURN DISTINCT collaborator.name AS name
            """,
                "hops": 2,
            },
            "What venues have the author '{}' published in?": {
                "cypher": """
            MATCH (author:Physics:author {{id: '{}'}})
            -[:paper]->(paper:Physics:paper)
            -[:venue]->(venue:Physics:venue)
            RETURN DISTINCT venue.name AS name
            """,
                "hops": 2,
            },
        },
        "paper": {
            "Who are the authors of the paper '{}'?": {
                "cypher": """
            MATCH (p:Physics:paper {{id: '{}'}})
            -[:author]->(a:Physics:author)
            RETURN DISTINCT a.name AS name
            """,
                "hops": 1,
            },
            "Where is the paper '{}' published?": {
                "cypher": """
            MATCH (p:Physics:paper {{id: '{}'}})
            -[:venue]->(v:Physics:venue)
            RETURN DISTINCT v.name AS name
            """,
                "hops": 1,
            },
            "Who are the academic collaborators of the author who writes the paper '{}'?": {
                "cypher": """
            MATCH (p:Physics:paper {{id: '{}'}})
            MATCH (p)-[:author]->(a:Physics:author)
            MATCH (a)-[:paper]->(otherPaper:Physics:paper)
            MATCH (otherPaper)-[:author]->(coAuthor:Physics:author)
            WHERE coAuthor <> a
            RETURN DISTINCT coAuthor.name AS name
            """,
                "hops": 3,
            },
            "What venues have the author of the paper '{}' published in?": {
                "cypher": """
            MATCH (p:Physics:paper {{id: '{}'}})
            -[:author]->(a:Physics:author)
            -[:paper]->(other_p:Physics:paper)
            -[:venue]->(v:Physics:venue)
            RETURN DISTINCT v.name AS name
            """,
                "hops": 3,
            },
            "What venues have the academic collaborators of the author who writes the paper '{}' published in?": {
                "cypher": """
            MATCH (start_paper:Physics:paper {{id: '{}'}})
            -[:author]->(author:Physics:author)
            -[:paper]->(collab_paper:Physics:paper)
            -[:author]->(collaborator:Physics:author)
            WHERE collaborator <> author
            MATCH (collaborator)-[:paper]->(pub:Physics:paper)
            -[:venue]->(venue:Physics:venue)
            RETURN DISTINCT venue.name AS name
            """,
                "hops": 5,
            },
        },
    },
    "amazon": {
        "item": {
            "What is the brand of the item '{}'?": {
                "cypher": """
            MATCH (:amazon:item {{id: '{}'}})
            -[:brand]->(brand:amazon:brand)
            RETURN DISTINCT brand.name as name
            """,
                "hops": 1,
            },
            "What are the brands of the items that are also brought after viewing the item '{}'?": {
                "cypher": """
            """,
                "hops": 2,
            },
            "What are the items that are also viewed when viewing items of the brand owning the item '{}'?": {
                "hops": 3,
            },
            "What are the brands of the items that are brought together with items of the brand owning the item '{}'?": {
                "hops": 4,
            },
        },
        "brand": {
            "What are the items of the brand '{}'?": {"hops": 1},
            "What are the items that are also brought together with items of the brand '{}'?": {
                "hops": 2,
            },
            "What are the brands of the items that are also brought after viewing items of the brand '{}'?": {
                "hops": 3,
            },
            "What items does the brands of the items that are also brought after viewing items of the brand '{}' have?": {
                "hops": 4
            },
        },
    },
    "goodreads": {
        "book": {
            "Who is the author of the book '{}'?": {
                "cypher": """
            MATCH (book:goodreads:book {{id: '{}'}})
            -[:author]->(author:goodreads:author)
            RETURN DISTINCT author.name as name
            """,
                "hops": 1,
            },
            "What series have the author of the book '{}' published?": {
                "cypher": """
            MATCH (book:goodreads:book {{id: '{}'}})
            MATCH (book)-[:author]->(author:goodreads:author)
            MATCH (author)-[:book]->(other_books:goodreads:book)
            MATCH (other_books)-[:series]->(series:goodreads:series)
            RETURN DISTINCT series.name as name
            """,
                "hops": 3,
            },
        },
        "author": {
            "What books has the author '{}' published?": {
                "cypher": """
            MATCH (author:goodreads:author {{id: '{}'}})
            -[:book]->(book:goodreads:book)
            RETURN DISTINCT book.name as name
            """,
                "hops": 1,
            },
            "What books have the collaborators of the author '{}' published?": {
                "cypher": """
            MATCH (author1:goodreads:author {{id: '{}'}})-[:book]->(book1:goodreads:book)
            MATCH (book1)-[:author]->(coauthor:goodreads:author)
            WHERE coauthor <> author1
            MATCH (coauthor)-[:book]->(other_book:goodreads:book)
            RETURN DISTINCT other_book.name as name
            """,
                "hops": 3,
            },
            "What are the series published by the publishers that have published books of the author '{}'?": {
                "cypher": """
            MATCH (author:goodreads:author {{id: '{}'}})
            -[:book]->(authorBook:goodreads:book)
            -[:publisher]->(publisher:goodreads:publisher)
            -[:book]->(publisherBook:goodreads:book)
            -[:series]->(series:goodreads:series)
            RETURN DISTINCT series.name as name
            """,
                "hops": 4,
            },
        },
        "publisher": {
            "What are the authors of the books published by the publisher '{}'?": {
                "cypher": """
            MATCH (:goodreads:publisher {{id: '{}'}})
            -[:book]->(b:goodreads:book)-[:author]->(a:goodreads:author)
            RETURN DISTINCT a.name as name
            """,
                "hops": 2,
            },
        },
        "series": {
            "Where does the books of the series '{}' published in?": {
                "cypher": """
            MATCH (s:goodreads:series {{id: '{}'}})
            MATCH (s)-[:book]->(b:goodreads:book)-[:publisher]->(p:goodreads:publisher)
            RETURN DISTINCT p.name as name
            """,
                "hops": 2,
            },
            "What are the authors of the books that are published by the publishers that have published books of the series '{}'?": {
                "cypher": """
            MATCH (s:goodreads:series {{id: '{}'}})
            -[:book]->(book1:goodreads:book)
            -[:publisher]->(p:goodreads:publisher)
            -[:book]->(book2:goodreads:book)
            -[:author]->(a:goodreads:author)
            RETURN DISTINCT a.name as name
            """,
                "hops": 4,
            },
        },
    },
}


multi_entity_concrete_template = {
    "physics": {
        "What is the relationship between authors '{}' and '{}' regarding collaborated papers?": {
            "cypher": """
            MATCH (author1:Physics:author)
            -[:paper]->(paper1:Physics:paper)
            -[:author]->(author2:Physics:author)
            WHERE author1 <> author2
            RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
            """,
            "hops": 2,
        },
        "What is the relationship between authors '{}' and '{}' regarding paper references?": {
            "cypher": """
            MATCH (author1:Physics:author)
            -[:paper]->(paper1:Physics:paper)
            -[:reference|cited_by]->(paper2:Physics:paper)
            -[:author]->(author2:Physics:author)
            WHERE author1 <> author2
            RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
            """,
            "hops": 3,
        },
        "What is the relationship between authors authors '{}' and '{}' regarding common collaborators?": {
            "cypher": """
            MATCH (author1:Physics:author)
            -[:paper]->(paper1:Physics:paper)
            -[:author]->(collaborator:Physics:author)
            -[:paper]->(paper2:Physics:paper)
            -[:author]->(author2:Physics:author)
            WHERE author1 <> collaborator AND author2 <> collaborator
            RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
            """,
            "hops": 4,
        },
        "What is the relationship between authors '{}' and '{}' regarding common venues they have published in?": {
            "cypher": """
            MATCH (author1:Physics:author)
            -[:paper]->(paper1:Physics:paper)
            -[:venue]->(venue:Physics:venue)
            -[:paper]->(paper2:Physics:paper)
            -[:author]->(author2:Physics:author)
            WHERE author1 <> author2
            RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
            """,
            "hops": 4,
        },
    },
    "amazon": {
        "What is the relationship between items '{}' and '{}' regarding common brands?": {
            "hops": 2
        },
        "What is the relationship between brands '{}' and '{}' regarding item purchasing?": {
            "hops": 3
        },
        "What is the relationship between items '{}' and '{}' regarding brands that are purchased together?": {
            "hops": 4
        },
        "What is the relationship between brands '{}' and '{}' regarding commonly viewed brands?": {
            "hops": 6
        },
    },
    "goodreads": {
        "What is the relationship between authors '{}' and '{}' regarding collaborated books?": {
            "cypher": """
        MATCH path = (author1:goodreads:author)-[:book]->(book:goodreads:book)<-[:book]-(author2:goodreads:author)
        WHERE author1 <> author2
        RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
        """,
            "hops": 2,
        },
        "What is the relationship between authors '{}' and '{}' regarding common series?": {
            "cypher": """
        MATCH path = (author1:goodreads:author)-[:book]->(book1:goodreads:book)-[:series]->(series:goodreads:series)<-[:series]-(book2:goodreads:book)<-[:book]-(author2:goodreads:author)
        WHERE author1 <> author2
        RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
        """,
            "hops": 4,
        },
        "What is the relationship between authors '{}' and '{}' regarding common publishers?": {
            "cypher": """
        MATCH path = (author1:goodreads:author)-[:book]->(book1:goodreads:book)-[:publisher]->(publisher:goodreads:publisher)<-[:publisher]-(book2:goodreads:book)<-[:book]-(author2:goodreads:author)
        WHERE author1 <> author2
        RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
        """,
            "hops": 4,
        },
        "What is the relationship between publishers '{}' and '{}' regarding commonly involved authors?": {
            "cypher": """
        MATCH path = (publisher1:goodreads:publisher)-[:book]->(book1:goodreads:book)-[:author]->(author:goodreads:author)-[:book]->(book2:goodreads:book)-[:publisher]->(publisher2:goodreads:publisher)
        WHERE publisher1 <> publisher2
        RETURN publisher1.name AS name1, publisher1.id AS id1, publisher2.name AS name2, publisher2.id AS id2
        """,
            "hops": 4,
        },
    },
}


def gen_single_entity_abstract(graph: nx.Graph, n: int, output_path: str):
    all_nodes = list(graph.nodes())
    questions = []
    i = 0
    name_set = {""}
    while i < n:
        node = random.choice(all_nodes)
        node_data = graph.nodes[node]
        if node_data["name"] in name_set:
            continue
        # if it has no neighbors, skip
        if graph.degree(node) < 10:
            continue
        question = f"Tell me about '{node_data['name']}'."
        q_entity = {
            "qid": i,
            "question": question,
            "entity": {node_data["name"]: node},
            "type": "single_entity_abstract",
            "hops": 1,
            "answer": "N/A",
        }
        questions.append(q_entity)
        name_set.add(node_data["name"])
        print(q_entity)
        i += 1
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
                if record["name1"] == record["name2"]:
                    continue
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

dataset_name = os.path.basename(args.path).lower()

# load the graph
graph = pickle.load(open(os.path.join(args.path, "graph.pkl"), "rb"))
print("NetworkX graph loaded")
print("# nodes:", graph.number_of_nodes())
print("# edges:", graph.number_of_edges())

# generate questions
gen_single_entity_abstract(
    graph, 80, os.path.join(args.path, "single_entity_abstract.jsonl")
)
gen_single_entity_concrete(
    graph, 10, dataset_name, os.path.join(args.path, "single_entity_concrete.jsonl")
)
gen_multi_entity_abstract(
    graph, 20, [2, 3, 4, 5], os.path.join(args.path, "multi_entity_abstract.jsonl")
)
gen_multi_entity_concrete(
    graph, 10, dataset_name, os.path.join(args.path, "multi_entity_concrete.jsonl")
)
