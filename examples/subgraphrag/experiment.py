import os
import random
import torch
import logging
import numpy as np
import argparse
import jsonlines
import networkx as nx
from datasets import load_dataset
from polyg import GraphRAG, QueryParam
from polyg.storage import Neo4jStorage
from neo4j import GraphDatabase
from tqdm import tqdm
from dotenv import load_dotenv
from retriever import subgraphrag_retriever
from dataloader import RetrieverDataset, collate_retriever
from model import Retriever
from prompts import icl_user_prompt, icl_ass_prompt

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
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
    ],
    required=True,
)
argparser.add_argument(
    "--benchmark", type=str, default="webqsp", choices=["webqsp", "cwq"], required=True
)
argparser.add_argument(
    "--path",
    type=str,
    required=True,
    help="Path to a saved model checkpoint, e.g., webqsp_Nov08-01:14:47/cpt.pth",
)
args = argparser.parse_args()
print(args)

RESULT_DIR = f"../results/{args.benchmark}/{args.model}"
MAX_MODEL_LEN = 65536
MAX_CONTEXT_TOKENS = 57344
MAX_OUTPUT_TOKENS = 8192

print(
    f"Benchmark dir: {args.benchmark}",
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


sampling_params = {}
if args.model in ["Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]:
    sampling_params = {
        "api_base": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        # Standard OpenAI parameters
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": MAX_OUTPUT_TOKENS,
        # vLLM-specific (or Qwen3-specific) parameters
        "top_k": 20,
        "min_p": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    lite_llm_model_name = "hosted_vllm/" + args.model
else:
    lite_llm_model_name = args.model
print(f"Sampling params: {sampling_params}")


rag = GraphRAG(
    dataset=args.benchmark,
    graph_storage_cls=Neo4jStorage,
    addon_params=neo4j_config,
    model=lite_llm_model_name,
    model_max_token_size=MAX_MODEL_LEN,
    model_sampling_params=sampling_params,
)
rag.register_retriever("subgraphrag", subgraphrag_retriever)


def print_outputs(outputs):
    print("=" * 80)
    print("Generated response:")
    print(outputs)
    print("-" * 80)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def subgraphrag(question, id_mapping, extra_data):
    print(f"Question: {question}")
    response, duration, token_len, api_calls, answer_list = rag.query(
        question,
        id_mapping,
        param=QueryParam(
            mode="local",
            edge_depth=1,
            local_context_length=MAX_CONTEXT_TOKENS,
            traversal_type="subgraphrag",
            response_type=(
                "Based on the relations retrieved from a knowledge graph, please answer the question."
                ' Please return formatted answers as a list, each prefixed with "ans:".'
                f" Example Question and Response: \n\n {icl_user_prompt} \n\n {icl_ass_prompt}"
            ),
            token_ratio_for_node=0.5,
            token_ratio_for_edge=0.4,
            failure_retries=0,
            extra_data=extra_data,
        ),
    )
    print_outputs(response)
    return "subgraphrag", response, duration, token_len, api_calls, answer_list


def build_graph(graph: list) -> nx.DiGraph:
    G = nx.DiGraph()
    for triplet in graph:
        h, r, t = triplet
        characters_to_replace = [".", "-", "#", " "]
        for char in characters_to_replace:
            r = r.replace(char, "_")
        G.add_edge(h, t, relation=r.strip())
    return G


def insert_to_neo4j(args: argparse.Namespace, nx_graph: nx.DiGraph):
    with driver.session() as session:
        # 1. Create Nodes
        for node_id, properties in tqdm(nx_graph.nodes(data=True), ncols=100):
            node_prop = {"name": node_id}
            # Ensure a label is set for the node
            query = (
                f"MERGE (n:{args.benchmark}:node {{id: $node_id}})"
                "SET n += $properties"
            )
            session.run(query, node_id=node_id, properties=node_prop)  # type: ignore

        # 2. Create Relationships
        for u, v, properties in tqdm(nx_graph.edges(data=True), ncols=100):
            # Ensure a relation is set
            rel_type = properties["relation"]

            query = (
                f"MATCH (a:{args.benchmark}:node {{id: $source_id}})"
                f"MATCH (b:{args.benchmark}:node {{id: $target_id}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "  # Using MERGE to avoid duplicate relationships
            )
            session.run(query, source_id=u, target_id=v)  # type: ignore

        # 3. Create indexes for faster lookup
        session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{args.benchmark}) ON (n.id)")  # type: ignore

    print("NetworkX graph successfully inserted into Neo4j.")


def remove_from_neo4j(args: argparse.Namespace):
    with driver.session() as session:
        session.run(f"MATCH (n:{args.benchmark}:node) DETACH DELETE n")  # type: ignore
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
    device = torch.device(f"cuda:0")

    cpt = torch.load(args.path, map_location="cpu")
    config = cpt["config"]
    set_seed(config["env"]["seed"])
    torch.set_num_threads(config["env"]["num_threads"])

    infer_set = RetrieverDataset(config=config, split="test", skip_no_path=False)

    emb_size = infer_set[0]["q_emb"].shape[-1]
    model = Retriever(emb_size, **config["retriever"]).to(device)
    model.load_state_dict(cpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    remove_from_neo4j(args)  # Clean up the Neo4j database after each sample

    output_file = os.path.join(RESULT_DIR, "results_subgraphrag.jsonl")

    dataset = load_dataset(f"rmanluo/RoG-{args.benchmark}", split="test")

    for it, sample in enumerate(dataset):
        sample_id = sample["id"]
        raw_sample = infer_set.get_by_id(sample_id)
        collate_sample = collate_retriever([raw_sample])

        question = sample["question"]
        nx_graph = build_graph(sample["graph"])
        id_mapping = {}
        for entity in sample["q_entity"]:
            id_mapping[entity] = entity
        insert_to_neo4j(args, nx_graph)  # Insert the graph into Neo4j
        rag.concrete_graph_schema = extract_graph_schema(nx_graph)  # Extract schema

        results = []
        results.append(
            subgraphrag(
                question,
                id_mapping.copy(),
                {
                    "model": model,
                    "sample": collate_sample,
                    "device": device,
                    "topk": 100,
                },
            )
        )

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
