import os
import logging
import numpy as np
from polyg import GraphRAG, QueryParam
from polyg._storage import HNSWVectorStorage
from polyg._utils import wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
from typing import List
import pickle
import torch

logging.basicConfig(level=logging.WARNING)
logging.getLogger("polyg").setLevel(logging.INFO)


DATASET_DIR = "datasets/maple/physics"
WORKING_DIR = f"./polyg_vllm_and_local_embedding_{DATASET_DIR.split('/')[-1]}"
MAX_MODEL_LEN = 15000
MAX_CONTEXT_TOKENS = 7000
MAX_OUTPUT_TOKENS = 5000

print("Dataset dir:", DATASET_DIR, "Working dir:", WORKING_DIR)

llm, sampling_params = None, None


def print_outputs(outputs):
    print("=" * 80)
    print("Generated reponse:")
    print(outputs)
    print("-" * 80)


async def vllm_llama_model(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[dict] = [],
    **kwargs,
) -> str:
    # openai_async_client = AsyncOpenAI(
    #     api_key=DEEPSEEK_API_KEY, base_url=API_BASE
    # )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # response = await openai_async_client.chat.completions.create(
    #     model=MODEL, messages=messages, **kwargs
    # )
    response = llm.chat(messages=messages, sampling_params=sampling_params)
    return response[0].outputs[0].text


async def vllm_llama_model_batch_inference(
    prompt: List[str],
    system_prompt: List[str] = None,
    history_messages: List[List[dict]] = [],
    **kwargs,
) -> str:
    batch_messages = []
    for i in range(len(prompt)):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt[i]})

        if history_messages:
            messages.extend(history_messages[i])
        messages.append({"role": "user", "content": prompt[i]})
        batch_messages.append(messages)
    print(batch_messages)
    batch_response = llm.chat(messages=batch_messages, sampling_params=sampling_params)
    return [response.outputs[0].text for response in batch_response]


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
    from time import time

    graph = pickle.load(open(os.path.join(DATASET_DIR, "graph.pkl"), "rb"))
    print("NetworkX graph loaded")

    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/entities_hnsw_metadata.pkl")
    remove_if_exist(f"{WORKING_DIR}/entities_hnsw.index")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.pkl")

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        model_func=vllm_llama_model,
        embedding_func=local_embedding,
        model_max_token_size=MAX_CONTEXT_TOKENS,
        vector_db_storage_cls=HNSWVectorStorage,
        vector_db_storage_cls_kwargs={
            "max_elements": 10000000,
            "ef_search": 200,
            "M": 50,
        },
        embedding_batch_num=2048,
    )
    start = time()
    rag.insert_from_graph(graph)
    print("indexing time:", time() - start, "s")


def query():
    global llm, sampling_params
    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        gpu_memory_utilization=0.95,
        max_model_len=MAX_MODEL_LEN,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=False,
        model_func=vllm_llama_model_batch_inference,
        embedding_func=local_embedding,
        model_max_token_size=MAX_MODEL_LEN,
        vector_db_storage_cls=HNSWVectorStorage,
        vector_db_storage_cls_kwargs={
            "max_elements": 10000000,
            "ef_search": 200,
            "M": 50,
        },
    )
    batch_response = rag.batch_query(
        [
            # 'What is the relationship between "B. C. S. Grandi" and "Wellington Figueiredo"?'
            # 'What is the relationship between "Julia Scharwächter" and "Miguel A. Pérez-Torres"?'
            'what is the relationship between "J. Koll" and "Z. Staykova"?'
            for _ in range(1)
        ],
        param=QueryParam(
            mode="local",
            # top_k=10,
            edge_depth=2,
            local_context_length=MAX_CONTEXT_TOKENS,
            traversal_type="shortest_path",
            response_type="a sentence or a paragraph based on provided information, concise while comprehensive about details.",
            local_token_ratio_for_node=0.6,
            local_token_ratio_for_edge=0.4,
        ),
    )
    for reponse in batch_response:
        print_outputs(reponse)


if __name__ == "__main__":
    # insert()
    query()
