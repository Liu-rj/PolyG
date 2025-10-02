import os
import logging
from datasets import load_dataset
import argparse
import jsonlines
import networkx as nx
from polyg import GraphRAG, QueryParam
from polyg.storage import Neo4jStorage
from neo4j import GraphDatabase
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logging.getLogger("polyg").setLevel(logging.INFO)


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--model",
    type=str,
    default="openai/gpt-4o",
    choices=[
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "deepseek/deepseek-chat",
        "deepseek/deepseek-reasoner",
    ],
    required=True,
)
argparser.add_argument(
    "--benchmark", type=str, default="webqsp", choices=["webqsp", "cwq"], required=True
)
args = argparser.parse_args()

DATASET_DIR = args.data_dir
DATASET_NAME = DATASET_DIR.split("/")[-1]
RESULT_DIR = f"results/{DATASET_NAME}/{args.model}"
MAX_MODEL_LEN = 128000
MAX_CONTEXT_TOKENS = 100000
MAX_OUTPUT_TOKENS = 5000

print(
    f"DATASET: {DATASET_NAME}",
    f"Dataset dir: {DATASET_DIR}",
    f"Benchmark dir: {args.benchmark_dir}",
    f"Result dir: {RESULT_DIR}",
)

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

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


def print_outputs(outputs):
    print("=" * 80)
    print("Generated reponse:")
    print(outputs)
    print("-" * 80)


rag = GraphRAG(
    dataset=DATASET_NAME,
    model=args.model,
    model_max_token_size=MAX_MODEL_LEN,
    graph_storage_cls=Neo4jStorage,
    addon_params=neo4j_config,
)


def BFS(question, id_mapping):
    print(f"Question: {question}")
    response, duration, token_len, api_calls, answer_list = rag.query(
        question,
        id_mapping,
        param=QueryParam(
            mode="local",
            edge_depth=1,
            local_context_length=MAX_CONTEXT_TOKENS,
            traversal_type="BFS",
            response_type="a simple sentence that indicates the answer.",
            local_token_ratio_for_node=0.6,
            local_token_ratio_for_edge=0.4,
            failure_retries=0,
        ),
    )
    print_outputs(response)
    return "BFS", response, duration, token_len, api_calls, answer_list


def cypher_single_entity(question, id_mapping):
    print(f"Question: {question}")
    response, duration, token_len, api_calls, answer_list = rag.query(
        question,
        id_mapping,
        param=QueryParam(
            mode="local",
            local_context_length=MAX_CONTEXT_TOKENS,
            traversal_type="cypher_query",
            response_type="a simple sentence that indicates the answer.",
            local_token_ratio_for_node=0.6,
            local_token_ratio_for_edge=0.4,
            failure_retries=0,
        ),
    )
    print_outputs(response)
    return "cypher_single_entity", response, duration, token_len, api_calls, answer_list


def cypher_only(question, id_mapping):
    print(f"Question: {question}")
    response, duration, token_len, api_calls, answer_list = rag.query(
        question,
        id_mapping,
        param=QueryParam(
            mode="local",
            local_context_length=MAX_CONTEXT_TOKENS,
            traversal_type="cypher_only",
            response_type="a simple sentence that indicates the answer.",
            local_token_ratio_for_node=0.6,
            local_token_ratio_for_edge=0.4,
            failure_retries=0,
        ),
    )
    print_outputs(response)
    return "cypher_only", response, duration, token_len, api_calls, answer_list


def adaptive(question, id_mapping):
    print(f"Question: {question}")
    query_param = QueryParam(
        mode="local",
        edge_depth=1,
        local_context_length=MAX_CONTEXT_TOKENS,
        traversal_type="cypher_query",
        response_type="a simple sentence that indicates the answer.",
        local_token_ratio_for_node=0.6,
        local_token_ratio_for_edge=0.4,
        failure_retries=3,
    )
    response, duration, token_len, api_calls, answer_list = rag.query(
        question,
        id_mapping,
        param=query_param,
    )
    print_outputs(response)
    return (
        "adaptive",
        response,
        duration,
        token_len,
        api_calls,
        answer_list,
        query_param.question_classification_result,
    )


def build_graph(graph: list) -> nx.DiGraph:
    G = nx.DiGraph()
    for triplet in graph:
        h, r, t = triplet
        characters_to_replace = [".", "-", "#", " "]
        for char in characters_to_replace:
            r = r.replace(char, "_")
        G.add_edge(h, t, relation=r.strip())
    return G


def upsert_to_neo4j(args: argparse.Namespace, nx_graph: nx.DiGraph):
    with driver.session() as session:
        # 1. Create Nodes
        for node_id, properties in tqdm(nx_graph.nodes(data=True), ncols=100):
            node_prop = {"name": node_id}
            # Ensure a label is set for the node
            query = (
                f"MERGE (n:{args.benchmark}:node {{id: $node_id}})"
                "SET n += $properties"
            )
            session.run(query, node_id=node_id, properties=node_prop)

        # 2. Create Relationships
        for u, v, properties in tqdm(nx_graph.edges(data=True), ncols=100):
            # Ensure a relation is set
            rel_type = properties["relation"]

            query = (
                f"MATCH (a:{args.benchmark}:node {{id: $source_id}})"
                f"MATCH (b:{args.benchmark}:node {{id: $target_id}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "  # Using MERGE to avoid duplicate relationships
            )
            session.run(query, source_id=u, target_id=v)

        # 3. Create indexes for faster lookup
        session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{args.benchmark}) ON (n.id)")

    print("NetworkX graph successfully inserted into Neo4j.")


def remove_from_neo4j(args: argparse.Namespace):
    with driver.session() as session:
        session.run(f"MATCH (n:{args.benchmark}:node) DETACH DELETE n")
    print(f"All nodes and relations in the {args.benchmark} graph have been removed.")


def extract_graph_schema(nx_graph: nx.DiGraph) -> str:
    all_relation_types = set()
    for u, v, properties in tqdm(nx_graph.edges(data=True), ncols=100):
        rel_type = properties["relation"]  # Default if not in properties
        all_relation_types.add(rel_type)

    print("Number of relation types:", len(all_relation_types))
    schema = ""
    for i, rel_type in enumerate(all_relation_types):
        schema += f"{i + 1}. {rel_type}\n"

    return schema


if __name__ == "__main__":
    remove_from_neo4j(args)  # Clean up the Neo4j database after each sample

    output_file = os.path.join(RESULT_DIR, "results_rephrased.jsonl")

    dataset = load_dataset(f"rmanluo/RoG-{args.benchmark}", split="test")
    # dataset = load_dataset("./datasets/cwq", split="test")

    # # randomly sample 100 examples
    # dataset = dataset.shuffle(seed=42).select(range(100))
    # with open("../datasets/cwq/test_ids.txt", "r") as f:
    #     test_ids = f.read().splitlines()
    # print(f"Number of test ids: {len(test_ids)}")
    # print(f"Number of unique test ids: {len(set(test_ids))}")

    for it, sample in enumerate(dataset):
        # if sample["id"] not in test_ids:
        #     continue
        question = sample["question"]
        nx_graph = build_graph(sample["graph"])
        id_mapping = {}
        for entity in sample["q_entity"]:
            id_mapping[entity] = entity
        upsert_to_neo4j(args, nx_graph)  # Insert the graph into Neo4j
        rag.concrete_graph_schema = extract_graph_schema(
            nx_graph
        )  # Extract schema if needed

        results = []
        results.append(adaptive(question, id_mapping.copy()))
        results.append(BFS(question, id_mapping.copy()))
        results.append(cypher_single_entity(question, id_mapping.copy()))
        results.append(cypher_only(question, id_mapping.copy()))

        result_entrees = []
        for result in results:
            result_entree = {
                "question_type": args.benchmark,
                "question": question,
                "method": result[0],
                "model_answer": result[1],
                "duration": round(result[2], 2),
                "token_count": result[3],
                "api_calls": result[4],
                "answer_list": result[5],
                "gt_answer": sample["a_entity"],
            }
            if len(result) > 6:
                result_entree["question_classification_result"] = result[6]
            result_entrees.append(result_entree)
            print(result_entree)

        remove_from_neo4j(args)  # Clean up the Neo4j database after each sample

        with jsonlines.open(output_file, "a") as writer:
            writer.write_all(result_entrees)

    driver.close()
