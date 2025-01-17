import os
import logging
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._storage import HNSWVectorStorage, Neo4jStorage
from nano_graphrag._utils import wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer
from typing import List
import torch
import boto3
import argparse
import jsonlines

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--data_dir", type=str, default="datasets/maple/Physics", required=True
)
argparser.add_argument(
    "--benchmark_dir", type=str, default="benchmarks/physics", required=True
)
args = argparser.parse_args()

DATASET_DIR = args.data_dir
WORKING_DIR = (
    f"checkpoints/nano_graphrag_bedrock_and_neo4j_{DATASET_DIR.split('/')[-1]}"
)
RESULT_DIR = f"results/{DATASET_DIR.split('/')[-1]}"
MAX_MODEL_LEN = 128000
MAX_CONTEXT_TOKENS = 100000
MAX_OUTPUT_TOKENS = 5000

print(
    f"Dataset dir: {DATASET_DIR}",
    f"Benchmark dir: {args.benchmark_dir}",
    f"Working dir: {WORKING_DIR}",
    f"Result dir: {RESULT_DIR}",
)

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

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


async def bedrock_generator(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[dict] = [],
    **kwargs,
) -> str:
    bedrock_cli = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
    )

    messages, system = [], []
    if system_prompt:
        system.append({"text": system_prompt})

    messages.append({"role": "user", "content": [{"text": prompt}]})

    response = bedrock_cli.converse(
        modelId=CHAT_MODEL_ID, messages=messages, system=system
    )
    return response["output"]["message"]["content"][0]["text"]


EMBEDDING_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", cache_folder=WORKING_DIR, device="cpu"
)


# We're using Sentence Transformers to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBEDDING_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBEDDING_MODEL.max_seq_length,
)
async def local_embedding(
    texts: list[str], batchsize: int = 32, device: torch.device = None
) -> np.ndarray:
    return EMBEDDING_MODEL.encode(
        texts, normalize_embeddings=True, batch_size=batchsize, device=device
    )


rag = GraphRAG(
    working_dir=WORKING_DIR,
    enable_llm_cache=False,
    best_model_func=bedrock_generator,
    cheap_model_func=bedrock_generator,
    embedding_func=local_embedding,
    best_model_max_token_size=MAX_MODEL_LEN,
    cheap_model_max_token_size=MAX_MODEL_LEN,
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
        ),
    )
    print_outputs(response)
    return "BFS", response, duration, token_len, api_calls, answer_list


def shortest_path(question, id_mapping):
    print(f"Question: {question}")
    response, duration, token_len, api_calls, answer_list = rag.query(
        question,
        id_mapping,
        param=QueryParam(
            mode="local",
            edge_depth=1,
            local_context_length=MAX_CONTEXT_TOKENS,
            traversal_type="all_shortest_paths",
            response_type="a sentence or a paragraph based on provided information, concise while comprehensive about details.",
            local_token_ratio_for_node=0.6,
            local_token_ratio_for_edge=0.4,
        ),
    )
    print_outputs(response)
    return "shortest_paths", response, duration, token_len, api_calls, answer_list


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
        ),
    )
    print_outputs(response)
    return "cypher_single_entity", response, duration, token_len, api_calls, answer_list


def cypher_multi_entity(question, id_mapping):
    print(f"Question: {question}")
    response, duration, token_len, api_calls, answer_list = rag.query(
        question,
        id_mapping,
        param=QueryParam(
            mode="local",
            local_context_length=MAX_CONTEXT_TOKENS,
            traversal_type="cypher_path_search",
            response_type="a sentence or a paragraph based on provided information, concise while comprehensive about details.",
            local_token_ratio_for_node=0.6,
            local_token_ratio_for_edge=0.4,
        ),
    )
    print_outputs(response)
    return "cypher_multi_entity", response, duration, token_len, api_calls, answer_list


if __name__ == "__main__":
    output_file = os.path.join(RESULT_DIR, "results.jsonl")
    # if os.path.exists(output_file):
    #     os.remove(output_file)

    question_types = [
        "single_entity_abstract",
        "single_entity_concrete",
        "multi_entity_abstract",
        "multi_entity_concrete",
    ]
    for question_type in question_types:
        contents = []
        with open(os.path.join(args.benchmark_dir, f"{question_type}.jsonl"), "r") as f:
            for item in jsonlines.Reader(f):
                contents.append(item)

        for item in contents:
            results = []
            question, id_mapping = item["question"], item["entity"]

            if question_type == "single_entity_abstract":
                results.append(BFS(question, id_mapping))
            elif question_type == "single_entity_concrete":
                results.append(BFS(question, id_mapping))
                results.append(cypher_single_entity(question, id_mapping))
            else:
                results.append(BFS(question, id_mapping))
                results.append(cypher_multi_entity(question, id_mapping))
                results.append(shortest_path(question, id_mapping))

            result_entrees = []
            for result in results:
                result_entree = {
                    "question_type": question_type,
                    "question": question,
                    "method": result[0],
                    "model_answer": result[1],
                    "duration": round(result[2], 2),
                    "token_count": result[3],
                    "api_calls": result[4],
                    "answer_list": result[5],
                    "gt_answer": item["answer"],
                }
                result_entrees.append(result_entree)
                print(result_entree)

            with jsonlines.open(output_file, "a") as writer:
                writer.write_all(result_entrees)
