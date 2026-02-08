import asyncio
import time
from typing import List, Tuple, Dict
from ..base import BaseGraphStorage, QueryParam, ID, RetrievalResult
from ..prompt import PROMPTS, SCHEMA_MAP
from ..utils import logger, num_tokens
from .utils import build_schema_example


async def guided_walk_retriever(
    query: str,
    id_mapping: dict[str, ID],
    kg_inst: BaseGraphStorage,
    query_param: QueryParam,
    global_config: dict,
) -> RetrievalResult:
    use_model_func = global_config["model_func"]

    tic = time.perf_counter()
    graph_schema = SCHEMA_MAP[global_config["dataset"]]
    if global_config["dataset"] in ["webqsp", "cwq"]:
        schema_example = await build_schema_example(kg_inst, list(id_mapping.values()))
        graph_schema = graph_schema.format(
            schema=global_config["concrete_graph_schema"],
            benchmark=global_config["dataset"],
            example=schema_example,
        )
    sys_prompt = PROMPTS["cypher_query_prompt"].format(graph_schema=graph_schema)

    token_len = 0
    history_msgs = []
    response = None
    llm_calls = 0

    # p_context, node_ids, dest_ids, ret_names = [], set(), set(), set()
    ret_ids = []
    cypher_query = ""
    prompt = f"query: {query}, id mapping: {id_mapping}"
    for i in range(query_param.failure_retries + 1):
        try:
            cur_tokens = num_tokens(sys_prompt + prompt, global_config["token_encoder"])
            token_len += cur_tokens
            llm_calls += 1
            response = await use_model_func(
                prompt=prompt,
                system_prompt=sys_prompt,
                history_messages=history_msgs,
            )
            print(response)
            cypher_query = response.split("```")[1].strip("cypher")
            print(f"Cypher query generation time: {time.perf_counter() - tic:.2f}s")
            print(f"Token length: {cur_tokens}")
            print("Generated cypher query:", cypher_query)

            tic = time.perf_counter()
            results = await kg_inst.exec_query(cypher_query)
            if len(results) == 0:
                raise ValueError("No result found, please adjust the query")
            ret_ids = [r["id"] for r in results]
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

    tic = time.perf_counter()
    aux_node_ids = set(id_mapping.values())
    unique_edges = set()
    if global_config["dataset"] in ["webqsp", "cwq"]:
        all_edges = await asyncio.gather(*[kg_inst.get_node_edges(d) for d in ret_ids])
        for edges in all_edges:
            hashable_edges = [frozenset(e.items()) for e in edges]
            unique_edges.update(hashable_edges)
        edges_as_dicts = [dict(e) for e in unique_edges]
        aux_node_ids.update([e["src_id"] for e in edges_as_dicts])
        aux_node_ids.update([e["tgt_id"] for e in edges_as_dicts])

    all_node_ids = list(aux_node_ids) + ret_ids
    nodes_data = await asyncio.gather(*[kg_inst.get_node(nid) for nid in all_node_ids])
    ret_names = [n["name"] for n in nodes_data[len(aux_node_ids) :]]  # type: ignore
    print(f"answer list: {ret_names}")
    print(f"Get node data time: {time.perf_counter() - tic:.2f}s")

    return RetrievalResult(
        cypher_query=cypher_query,
        nodes_data=nodes_data,  # type: ignore
        edges_data=[dict(x) for x in unique_edges],
        reasoning_paths=[],
        auxiliary_data=[],
        used_tokens=token_len,
        num_llm_calls=llm_calls,
        answer_list=ret_names,
    )
