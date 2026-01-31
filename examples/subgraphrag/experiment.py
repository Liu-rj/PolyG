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
                'Please return formatted answers by listing each answer on a separate line, starting with the prefix "ans:".'
            ),
            token_ratio_for_node=0,
            token_ratio_for_edge=0.9,
            failure_retries=0,
            extra_data=extra_data,
        ),
    )
    print_outputs(response)
    return "subgraphrag", response, duration, token_len, api_calls, answer_list


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

    output_file = os.path.join(RESULT_DIR, f"results_dc_{args.topk}.jsonl")

    # resume from history
    start_id = 0
    if os.path.exists(output_file):
        with jsonlines.open(output_file) as reader:
            done_lines = list(reader)
        questions = set([q["question"] for q in done_lines])
        done_count = len(questions)
        print(f"Resuming from {done_count} done questions.")
        start_id = done_count

    for it in range(start_id, len(infer_set)):
        sample = infer_set[it]
        collate_sample = collate_retriever([sample])

        question = sample["question"]
        id_mapping = {}
        for entity in sample["q_entity"]:
            id_mapping[entity] = entity

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

        with jsonlines.open(output_file, "a") as writer:
            writer.write_all(result_entrees)

    driver.close()
