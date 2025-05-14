import os
import logging
import numpy as np
import torch
import boto3
import argparse
import jsonlines
import time
from openai import OpenAI
from polyg import GraphRAG, QueryParam
from polyg._storage import HNSWVectorStorage, Neo4jStorage
from polyg._utils import wrap_embedding_func_with_attrs
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
        "claude-3.5-sonnet",
        "deepseek-chat",
        "llama-3.1-70b",
    ],
    required=True,
)
argparser.add_argument(
    "--data_dir", type=str, default="datasets/maple/Physics", required=True
)
argparser.add_argument(
    "--benchmark_dir", type=str, default="benchmarks/physics", required=True
)
args = argparser.parse_args()

DATASET_DIR = args.data_dir
WORKING_DIR = f"checkpoints/polyg_bedrock_and_neo4j_{DATASET_DIR.split('/')[-1]}"
RESULT_DIR = f"results/{DATASET_DIR.split('/')[-1]}/{args.model}"
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

neo4j_config = {
    "neo4j_url": os.environ.get("NEO4J_URL", "neo4j://localhost:7687"),
    "neo4j_auth": (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "12345678"),
    ),
}


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


def print_outputs(outputs):
    print("=" * 80)
    print("Generated reponse:")
    print(outputs)
    print("-" * 80)


async def bedrock_generator(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[Tuple] = [],
    **kwargs,
) -> str:
    messages, system = [], []
    if system_prompt:
        system.append({"text": system_prompt})

    history_messages = [
        {"role": r, "content": [{"text": m}]} for r, m in history_messages
    ]

    messages.extend(history_messages)
    messages.append({"role": "user", "content": [{"text": prompt}]})

    response = bedrock_cli.converse(modelId=MODEL_ID, messages=messages, system=system)
    return response["output"]["message"]["content"][0]["text"]


async def openai_generator(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[dict] = [],
    **kwargs,
) -> str:
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


if args.model in ["gpt-4o", "gpt-4o-mini"]:
    client = OpenAI()
    generator = openai_generator
elif args.model == "deepseek-chat":
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
    )
    generator = openai_generator
elif args.model == "claude-3.5-sonnet":
    bedrock_cli = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
    MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    generator = bedrock_generator
elif args.model == "llama-3.1-70b":
    bedrock_cli = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
    MODEL_ID = "meta.llama3-1-70b-instruct-v1:0"
    generator = bedrock_generator
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
            traversal_type="direct_cypher",
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
        failure_retries=0,
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
    output_file = os.path.join(RESULT_DIR, "results_rephrased_retry_0.jsonl")
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
        if question_type == "single_entity_concrete_rephrased":
            contents = contents[30:]

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
