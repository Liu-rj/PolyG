import asyncio
import time
from typing import List, Tuple, Dict
from .utils import build_schema_example
from ..base import BaseGraphStorage, QueryParam, ID, RetrievalResult
from ..prompt import PROMPTS, SCHEMA_MAP
from ..utils import logger, num_tokens


async def cypher_only_retriever(
    query: str,
    id_mapping: dict[str, ID],
    kg_inst: BaseGraphStorage,
    query_param: QueryParam,
    global_config: dict,
) -> RetrievalResult:
    use_model_func = global_config["model_func"]

    graph_schema = SCHEMA_MAP[global_config["dataset"]]
    if global_config["dataset"] in ["webqsp", "cwq"]:
        schema_example = await build_schema_example(kg_inst, list(id_mapping.values()))
        graph_schema = graph_schema.format(
            schema=global_config["concrete_graph_schema"],
            benchmark=global_config["dataset"],
            example=schema_example,
        )
    sys_prompt = PROMPTS["cypher_only_query"].format(graph_schema=graph_schema)

    token_len = 0
    history_msgs = []
    response = None
    llm_calls = 0

    cypher_query = ""
    results = []
    prompt = f"query: {query}, id mapping: {id_mapping}"
    for i in range(query_param.failure_retries + 1):
        try:
            tic = time.perf_counter()
            cur_tokens = num_tokens(sys_prompt + prompt, global_config["token_encoder"])
            token_len += cur_tokens
            llm_calls += 1
            response = await use_model_func(
                prompt=prompt,
                system_prompt=sys_prompt,
                history_messages=history_msgs,
            )
            cypher_query = response.split("```")[1].strip("cypher")
            print(f"Cypher query generation time: {time.perf_counter() - tic:.2f}s")
            print(f"Token length: {cur_tokens}")
            print("Generated cypher query:", cypher_query)

            tic = time.perf_counter()
            results = await kg_inst.exec_query(cypher_query)
            print(f"Query execution time: {time.perf_counter() - tic:.2f}s")

            break
        except Exception as e:
            logger.error(f"Error: {e}")
            history_msgs.extend(
                [
                    ("user", prompt),
                    ("assistant", response),
                    ("user", PROMPTS["error_retry"].format(str(e))),
                ]
            )

    return RetrievalResult(
        cypher_query=cypher_query,
        nodes_data=[],
        edges_data=[],
        reasoning_paths=[],
        auxiliary_data=results,
        used_tokens=token_len,
        num_llm_calls=llm_calls,
        answer_list=[],
    )
