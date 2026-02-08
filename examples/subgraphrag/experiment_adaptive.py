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
from prepare_data import get_data

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
argparser.add_argument("--topk", type=int, default=100)
args = argparser.parse_args()
print(args)

RESULT_DIR = f"results/{args.benchmark}/{args.model}"
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
print(rag.traversal_functions)


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
    query_param = QueryParam(
        mode="local",
        edge_depth=1,
        local_context_length=MAX_CONTEXT_TOKENS,
        traversal_type="subgraphrag",
        response_type=(
            'Please return formatted answers by listing each answer on a separate line, starting with the prefix "ans:".'
        ),
        token_ratio_for_node=0,
        token_ratio_for_edge=0.9,
        failure_retries=0,
        extra_data=extra_data,
    )
    response, duration, token_len, api_calls, answer_list = rag.query(
        question, id_mapping, param=query_param
    )
    if (
        "ans:" not in response.lower()
        or "ans: unknown" in response.lower()
        or ("ans: not " in response.lower() and response.lower().count("ans:") == 1)
        or ("ans: no " in response.lower() and response.lower().count("ans:") == 1)
    ):
        query_param = QueryParam(
            mode="local",
            edge_depth=1,
            local_context_length=MAX_CONTEXT_TOKENS,
            traversal_type="adaptive",
            response_type=(
                'Please return formatted answers by listing each answer on a separate line, starting with the prefix "ans:".'
            ),
            token_ratio_for_node=0.5,
            token_ratio_for_edge=0.4,
            failure_retries=3,
            extra_data=extra_data,
        )
        response, duration2, token_len2, api_calls2, answer_list2 = rag.query(
            question, id_mapping, param=query_param
        )
        duration += duration2
        token_len += token_len2
        api_calls += api_calls2
        answer_list += answer_list2
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
    remove_from_neo4j(args)  # Clean up the Neo4j database after each sample

    device = torch.device(f"cuda:0")

    cpt = torch.load(args.path, map_location="cpu")
    config = cpt["config"]
    set_seed(config["env"]["seed"])
    torch.set_num_threads(config["env"]["num_threads"])

    infer_set = RetrieverDataset(config=config, split="test", skip_no_path=False)
    dataset = load_dataset(f"rmanluo/RoG-{args.benchmark}", split="test")

    emb_size = infer_set[0]["q_emb"].shape[-1]
    model = Retriever(emb_size, **config["retriever"]).to(device)
    model.load_state_dict(cpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    output_file = os.path.join(
        RESULT_DIR, f"results_dc_{args.topk}_subgraphrag+adaptive.jsonl"
    )

    previous_answer_file = f"results/{args.benchmark}/{args.model}/results_dc_100.jsonl"
    answers = []
    with open(previous_answer_file, "r") as f:
        for item in jsonlines.Reader(f):
            answers.append(item)

    # resume from history
    start_id = 0
    if os.path.exists(output_file):
        with jsonlines.open(output_file) as reader:
            done_lines = list(reader)
        questions = set([q["question"] for q in done_lines])
        done_count = len(questions)
        print(f"Resuming from {done_count} done questions.")
        dataset = dataset.select(range(done_count, len(dataset)))  # type: ignore
        start_id = done_count

    for it, raw_sample in tqdm(enumerate(dataset, start=start_id)):
        # for it in tqdm(range(start_id, len(infer_set))):
        sample = infer_set[it]
        previous_answer = answers[it]
        assert sample["id"] == raw_sample["id"]
        assert sample["question"] == previous_answer["question"]

        answer = previous_answer["model_answer"]
        question = sample["question"]

        if not (
            "ans:" not in answer.lower()
            or "ans: unknown" in answer.lower()
            or ("ans: not " in answer.lower() and answer.lower().count("ans:") == 1)
            or ("ans: no " in answer.lower() and answer.lower().count("ans:") == 1)
        ):
            previous_answer["method"] = "adaptive"
            result_entrees = [previous_answer]
        else:
            collate_sample = collate_retriever([sample])

            id_mapping = {}
            for entity in sample["q_entity"]:
                id_mapping[entity] = entity

            nx_graph = build_graph(raw_sample["graph"])
            insert_to_neo4j(args, nx_graph)  # Insert the graph into Neo4j
            rag.concrete_graph_schema = extract_graph_schema(nx_graph)  # Extract schema

            results = []
            try:
                results.append(
                    subgraphrag(
                        question,
                        id_mapping.copy(),
                        {
                            "model": model,
                            "sample": collate_sample,
                            "device": device,
                            "topk": args.topk,
                            "maxk": 500,
                        },
                    )
                )
            except Exception as e:
                print(f"Error processing sample {it}: {e}")
                continue

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
