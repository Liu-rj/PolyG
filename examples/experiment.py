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
argparser.add_argument(
    "--benchmark_dir", type=str, default="../benchmarks/physics", required=True
)
args = argparser.parse_args()

DATASET_DIR = args.data_dir
WORKING_DIR = f"checkpoints/polyg_{DATASET_DIR.split('/')[-1]}"
RESULT_DIR = f"results/{DATASET_DIR.split('/')[-1]}/{args.model}"
MAX_MODEL_LEN = 128000
MAX_CONTEXT_TOKENS = 90000
MAX_OUTPUT_TOKENS = 5000

print(
    f"Dataset dir: {DATASET_DIR}",
    f"Benchmark dir: {args.benchmark_dir}",
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


def print_outputs(outputs):
    print("=" * 80)
    print("Generated reponse:")
    print(outputs)
    print("-" * 80)


rag = GraphRAG(
    working_dir=WORKING_DIR,
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


if __name__ == "__main__":
    output_file = os.path.join(RESULT_DIR, "results_rephrased_test.jsonl")
    # if os.path.exists(output_file):
    #     os.remove(output_file)

    question_types = [
        "single_entity_abstract_rephrased",
        "single_entity_concrete_rephrased",
        "multi_entity_abstract_rephrased",
        "multi_entity_concrete_rephrased",
        "nested_question_rephrased",
    ]
    for question_type in question_types:
        contents = []
        with open(os.path.join(args.benchmark_dir, f"{question_type}.jsonl"), "r") as f:
            for item in jsonlines.Reader(f):
                contents.append(item)

        contents = contents[0:1]

        for item in contents:
            results = []
            question, id_mapping = item["question"], item["entity"]

            results.append(adaptive(question, id_mapping.copy()))
            results.append(BFS(question, id_mapping.copy()))
            results.append(cypher_single_entity(question, id_mapping.copy()))
            results.append(cypher_only(question, id_mapping.copy()))

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
                if len(result) > 6:
                    result_entree["question_classification_result"] = result[6]
                result_entrees.append(result_entree)
                print(result_entree)

            with jsonlines.open(output_file, "a") as writer:
                writer.write_all(result_entrees)
