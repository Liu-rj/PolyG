import os
import logging
import boto3
import argparse
import jsonlines
from openai import OpenAI
from polyg import GraphRAG, QueryParam
from polyg.storage import Neo4jStorage
from typing import List, Tuple
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
    "--data_dir", type=str, default="../datasets/physics", required=True
)
args = argparser.parse_args()
print(args)

DATASET_DIR = args.data_dir
DATASET_NAME = DATASET_DIR.split("/")[-1]
RESULT_DIR = f"results/{DATASET_NAME}/{args.model}"
MAX_MODEL_LEN = 128000
MAX_CONTEXT_TOKENS = 90000
MAX_OUTPUT_TOKENS = 5000

print(
    f"DATASET: {DATASET_NAME}",
    f"Dataset dir: {DATASET_DIR}",
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


if __name__ == "__main__":
    question = "What are the key aspects and characteristics of 'josephson effect in mgb2 break junctions'?"
    id_mapping = {"josephson effect in mgb2 break junctions": "1618286189"}
    print(f"Question: {question}")
    query_param = QueryParam(
        mode="local",
        edge_depth=1,
        local_context_length=MAX_CONTEXT_TOKENS,
        traversal_type="adaptive",
        response_type="a sentence or a paragraph based on provided information, concise while comprehensive about details.",
        token_ratio_for_node=0.5,
        token_ratio_for_edge=0.4,
        failure_retries=3,
    )
    response, duration, token_len, api_calls, answer_list = rag.query(
        question,
        id_mapping,
        param=query_param,
    )
    print_outputs(response)
    print(
        "adaptive",
        duration,
        token_len,
        api_calls,
        answer_list,
        query_param.question_classification_result,
    )
