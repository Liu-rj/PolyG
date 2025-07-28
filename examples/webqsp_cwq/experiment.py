import os
import logging
import numpy as np
from datasets import load_dataset
import boto3
import argparse
import jsonlines
import networkx as nx
from openai import OpenAI
from polyg import GraphRAG, QueryParam
from polyg._storage import HNSWVectorStorage, Neo4jStorage
from polyg._utils import wrap_embedding_func_with_attrs
from neo4j import GraphDatabase
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logging.getLogger("polyg").setLevel(logging.INFO)


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--model",
    type=str,
    default="gpt-4o-mini",
    choices=[
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "claude-3.5-sonnet",
        "deepseek-chat",
        "deepseek-r1",
        "llama-3.1-405b",
        "mistral-large",
        "gemini-2.5-flash",
    ],
    required=True,
)
argparser.add_argument(
    "--benchmark", type=str, default="webqsp", choices=["webqsp", "cwq"], required=True
)
args = argparser.parse_args()

WORKING_DIR = f"checkpoints/polyg_bedrock_and_neo4j_{args.benchmark}"
RESULT_DIR = f"results/{args.benchmark}/{args.model}"
MAX_MODEL_LEN = 128000
MAX_CONTEXT_TOKENS = 90000
MAX_OUTPUT_TOKENS = 5000

print(
    f"Benchmark: {args.benchmark}",
    f"Working dir: {WORKING_DIR}",
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


EMBEDDING_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", cache_folder=WORKING_DIR, device="cpu"
)


# We're using Sentence Transformers to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBEDDING_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBEDDING_MODEL.max_seq_length,
)
async def local_embedding(
    texts: list[str], batchsize: int = 32, device: str | None = None
) -> np.ndarray:
    return EMBEDDING_MODEL.encode(
        sentences=texts, batch_size=batchsize, device=device, normalize_embeddings=True
    )


def print_outputs(outputs):
    print("=" * 80)
    print("Generated reponse:")
    print(outputs)
    print("-" * 80)


async def bedrock_generator(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: List[Tuple] = [],
    **kwargs,
) -> str:
    messages, system = [], []
    if system_prompt:
        system.append({"text": system_prompt})

    history_messages_formated = [
        {"role": r, "content": [{"text": m}]} for r, m in history_messages
    ]

    messages.extend(history_messages_formated)
    messages.append({"role": "user", "content": [{"text": prompt}]})

    response = bedrock_cli.converse(modelId=MODEL_ID, messages=messages, system=system)
    return response["output"]["message"]["content"][0]["text"]


async def openai_generator(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: List[dict] = [],
    **kwargs,
) -> str | None:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    history_messages = [{"role": r, "content": m} for r, m in history_messages]

    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=args.model, messages=messages, stream=False
    )
    return response.choices[0].message.content


if args.model in ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"]:
    client = OpenAI()
    generator = openai_generator
elif args.model == "deepseek-chat":
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
    )
    generator = openai_generator
elif args.model == "deepseek-r1":
    bedrock_cli = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
    MODEL_ID = "us.deepseek.r1-v1:0"
    generator = bedrock_generator
elif args.model == "claude-3.5-sonnet":
    bedrock_cli = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
    MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    generator = bedrock_generator
elif args.model == "llama-3.1-405b":
    bedrock_cli = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
    MODEL_ID = "meta.llama3-1-405b-instruct-v1:0"
    generator = bedrock_generator
elif args.model == "mistral-large":
    bedrock_cli = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
    MODEL_ID = "mistral.mistral-large-2407-v1:0"
    generator = bedrock_generator
elif args.model == "gemini-2.5-flash":
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    generator = openai_generator
else:
    raise ValueError(f"Unsupported model: {args.model}")


rag = GraphRAG(
    working_dir=WORKING_DIR,
    enable_llm_cache=False,
    model_func=generator,
    embedding_func=local_embedding,
    model_max_token_size=MAX_MODEL_LEN,
    vector_db_storage_cls=HNSWVectorStorage,
    graph_storage_cls=Neo4jStorage,
    addon_params=neo4j_config,
    vector_db_storage_cls_kwargs={
        "max_elements": 10000000,
        "ef_search": 200,
        "M": 50,
    },
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
            response_type="a sentence or a paragraph based on provided information, concise while comprehensive about details.",
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
            response_type="a sentence or a paragraph based on provided information, concise while comprehensive about details.",
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
            response_type="a sentence or a paragraph based on provided information, concise while comprehensive about details.",
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
        traversal_type="adaptive",
        response_type="a sentence or a paragraph based on provided information, concise while comprehensive about details.",
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

    for sample in dataset:
        question = sample["question"]
        nx_graph = build_graph(sample["graph"])
        id_mapping = {}
        for entity in sample["q_entity"]:
            id_mapping[entity] = entity
        upsert_to_neo4j(args, nx_graph)  # Insert the graph into Neo4j
        rag.graph_schema = extract_graph_schema(nx_graph)  # Extract schema if needed

        results = []
        results.append(adaptive(question, id_mapping.copy()))
        # results.append(BFS(question, id_mapping.copy()))
        # results.append(cypher_single_entity(question, id_mapping.copy()))
        # results.append(cypher_only(question, id_mapping.copy()))

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
