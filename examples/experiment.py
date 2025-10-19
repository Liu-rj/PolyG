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
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
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
print(args)

DATASET_DIR = args.data_dir
DATASET_NAME = DATASET_DIR.split("/")[-1]
RESULT_DIR = f"results/{DATASET_NAME}/{args.model}"
MAX_MODEL_LEN = 65536
MAX_CONTEXT_TOKENS = 57344
MAX_OUTPUT_TOKENS = 8192

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


def print_outputs(outputs):
    print("=" * 80)
    print("Generated reponse:")
    print(outputs)
    print("-" * 80)


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
elif args.model == "Qwen/Qwen3-Next-80B-A3B-Instruct":
    sampling_params = {
        "api_base": "http://localhost:10021/v1",
        "api_key": "EMPTY",
        # Standard OpenAI parameters
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": MAX_OUTPUT_TOKENS,
        # vLLM-specific (or Qwen3-specific) parameters
        "top_k": 20,
        "min_p": 0.0,
    }
    lite_llm_model_name = "openai/" + args.model
else:
    lite_llm_model_name = args.model
print(f"Sampling params: {sampling_params}")


rag = GraphRAG(
    dataset=DATASET_NAME,
    graph_storage_cls=Neo4jStorage,
    addon_params=neo4j_config,
    model=lite_llm_model_name,
    model_max_token_size=MAX_MODEL_LEN,
    model_sampling_params=sampling_params,
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
            token_ratio_for_node=0.5,
            token_ratio_for_edge=0.4,
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
            token_ratio_for_node=0.5,
            token_ratio_for_edge=0.4,
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
            token_ratio_for_node=0.5,
            token_ratio_for_edge=0.4,
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
    return (
        "adaptive",
        response,
        duration,
        token_len,
        api_calls,
        answer_list,
        query_param.question_classification_result,
    )


def BFS_PPR(question, id_mapping):
    print(f"Question: {question}")
    query_param = QueryParam(
        mode="local",
        edge_depth=2,
        local_context_length=MAX_CONTEXT_TOKENS,
        traversal_type="BFS+PPR",
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
    return (
        "BFS+PPR",
        response,
        duration,
        token_len,
        api_calls,
        answer_list,
        query_param.question_classification_result,
    )


if __name__ == "__main__":
    output_file = os.path.join(RESULT_DIR, "results_bfs_ppr.jsonl")
    # if os.path.exists(output_file):
    #     os.remove(output_file)

    question_types = [
        "single_entity_abstract",
        "single_entity_concrete",
        "multi_entity_abstract",
        "multi_entity_concrete",
        "nested_question",
    ]
    for question_type in question_types:
        contents = []
        with open(os.path.join(args.benchmark_dir, f"{question_type}.jsonl"), "r") as f:
            for item in jsonlines.Reader(f):
                contents.append(item)

        for item in contents:
            results = []
            question, id_mapping = item["question"], item["entity"]

            results.append(adaptive(question, id_mapping.copy()))
            results.append(BFS(question, id_mapping.copy()))
            results.append(cypher_single_entity(question, id_mapping.copy()))
            results.append(cypher_only(question, id_mapping.copy()))
            results.append(BFS_PPR(question, id_mapping.copy()))

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
