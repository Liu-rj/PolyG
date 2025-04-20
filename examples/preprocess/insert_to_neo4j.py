import os
import logging
import numpy as np
from polyg import GraphRAG
from polyg._storage import HNSWVectorStorage, Neo4jStorage
from polyg._utils import wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer
from typing import List
from time import time
import pickle
import torch
import boto3
import argparse

logging.basicConfig(level=logging.WARNING)
logging.getLogger("polyg").setLevel(logging.INFO)


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--path", type=str, default="datasets/maple/Physics", required=True
)
args = argparser.parse_args()

DATASET_DIR = args.path
WORKING_DIR = f"../checkpoints/polyg_bedrock_and_neo4j_{DATASET_DIR.split('/')[-1]}"
MAX_MODEL_LEN = 128000
MAX_CONTEXT_TOKENS = 100000
MAX_OUTPUT_TOKENS = 5000

print("Dataset dir:", DATASET_DIR, "Working dir:", WORKING_DIR)

CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

neo4j_config = {
    "neo4j_url": os.environ.get("NEO4J_URL", "neo4j://localhost:7687"),
    "neo4j_auth": (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "123456789"),
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


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


def insert():
    graph = pickle.load(open(os.path.join(DATASET_DIR, "graph.pkl"), "rb"))
    print("NetworkX graph loaded")
    print("# nodes:", graph.number_of_nodes())
    print("# edges:", graph.number_of_edges())

    # remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    # remove_if_exist(f"{WORKING_DIR}/entities_hnsw_metadata.pkl")
    # remove_if_exist(f"{WORKING_DIR}/entities_hnsw.index")

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        model_func=bedrock_generator,
        cheap_model_func=bedrock_generator,
        embedding_func=local_embedding,
        model_max_token_size=MAX_CONTEXT_TOKENS,
        cheap_model_max_token_size=MAX_CONTEXT_TOKENS,
        vector_db_storage_cls=HNSWVectorStorage,
        graph_storage_cls=Neo4jStorage,
        addon_params=neo4j_config,
        vector_db_storage_cls_kwargs={
            "max_elements": 10000000,
            "ef_search": 200,
            "M": 50,
        },
        embedding_batch_num=2048,
    )
    start = time()
    rag.insert_from_networkx_to_neo4j(graph)
    print("indexing time:", time() - start, "s")


if __name__ == "__main__":
    insert()
