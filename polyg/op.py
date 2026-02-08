import asyncio
import time
import networkx as nx
import tiktoken
import transformers
from typing import Union, List, Tuple, Dict, Callable
from .base import BaseGraphStorage, QueryParam, ID, RetrievalResult
from .prompt import (
    PROMPTS,
    SCHEMA_MAP,
    icl_sys_prompt,
    icl_ass_prompt,
    icl_user_prompt,
    icl_cot_prompt,
)
from .utils import (
    logger,
    truncate_list_by_token_size,
    num_tokens,
    enclose_string_with_quotes,
    list_to_csv,
)


ALL_CONTEXT = """
Cypher Query:
```cypher
{cypher_query}
```

Entities:
```csv
{entities_context}
```

Relations:
```csv
{relations_context}
```

Reasoning Paths:
{reasoning_path_context}

Auxiliary Data:
{auxdata_context}
"""

SUBGRAPHRAG_CONTEXT = """
Triplets:
{relations_context}
"""


async def sort_entity_relation(
    nodes_data: List[Dict], edges_data: List[Dict], kg_inst: BaseGraphStorage
) -> Tuple[List[Dict], List[Dict]]:
    tic = time.perf_counter()
    nodes_degree = await asyncio.gather(
        *[kg_inst.node_degree(n["id"]) for n in nodes_data]
    )
    nodes_data = [{**n, "rank": d} for n, d in zip(nodes_data, nodes_degree)]  # type: ignore
    nodes_data = sorted(nodes_data, key=lambda x: x["rank"], reverse=True)
    nodes_data_map = {n["id"]: n for n in nodes_data}
    logger.info(f"Get node data time: {time.perf_counter() - tic:.2f}s")

    tic = time.perf_counter()
    edges_nid = set()
    for e in edges_data:
        if e["src_id"] not in nodes_data_map:
            edges_nid.add(e["src_id"])
        if e["tgt_id"] not in nodes_data_map:
            edges_nid.add(e["tgt_id"])
    edges_nid = list(edges_nid)
    edge_node_data = await asyncio.gather(*[kg_inst.get_node(nid) for nid in edges_nid])
    for it, ndata in enumerate(edge_node_data):
        if ndata is None:
            print(edges_nid[it])
    edge_node_degree = await asyncio.gather(
        *[kg_inst.node_degree(nid) for nid in edges_nid]
    )
    edge_node_data = [{**n, "rank": d} for n, d in zip(edge_node_data, edge_node_degree)]  # type: ignore
    nodes_data_map.update({n["id"]: n for n in edge_node_data})
    logger.info(f"Get edge node data time: {time.perf_counter() - tic:.2f}s")

    tic = time.perf_counter()
    edges_name = [
        (nodes_data_map[e["src_id"]]["name"], nodes_data_map[e["tgt_id"]]["name"])
        for e in edges_data
    ]
    logger.info(f"Get edges name time: {time.perf_counter() - tic:.2f}s")

    tic = time.perf_counter()
    edges_degree = [
        nodes_data_map[e["src_id"]]["rank"] + nodes_data_map[e["tgt_id"]]["rank"]
        for e in edges_data
    ]
    logger.info(f"Get edge degree time: {time.perf_counter() - tic:.2f}s")

    tic = time.perf_counter()
    edges_data = [
        {"src_tgt": k, "rank": r, **v}
        for k, r, v in zip(edges_name, edges_degree, edges_data)
    ]
    logger.info(f"Combine edge data time: {time.perf_counter() - tic:.2f}s")

    tic = time.perf_counter()
    edges_data = sorted(edges_data, key=lambda x: x["rank"], reverse=True)
    logger.info(f"Sort edge data time: {time.perf_counter() - tic:.2f}s")
    return nodes_data, edges_data


def form_entity_relation_context(
    cypher_query: str,
    nodes_data: List[Dict],
    edges_data: List[Dict],
    reasoning_path: List,
    aux_data: List[Dict],
    query_param: QueryParam,
    token_encoder: tiktoken.Encoding | transformers.PreTrainedTokenizer,
) -> str:
    # build entity context
    tic = time.perf_counter()
    keys = ["name", "node_type", "description"]
    entity_header = ",\t".join([f"{enclose_string_with_quotes(data)}" for data in keys])
    entites_section_list = [entity_header]
    for i, n in enumerate(nodes_data):
        raw_data = [n.get(k, "UNKNOWN") for k in keys]
        entites_section_list.append(
            ",\t".join([f"{enclose_string_with_quotes(data)}" for data in raw_data])
        )
    truncated_entities_list = truncate_list_by_token_size(
        entites_section_list,
        max_token_size=int(
            query_param.local_context_length * query_param.token_ratio_for_node
        ),
        token_encoder=token_encoder,
    )
    if len(truncated_entities_list) == 0:
        entities_context = "No entities."
    else:
        entities_context = list_to_csv(truncated_entities_list)
    print(f"Form entity context time: {time.perf_counter() - tic:.2f}s")

    # build relation context
    tic = time.perf_counter()
    if query_param.traversal_type == "subgraphrag":
        relations_section_list = []
        for i, e in enumerate(edges_data):
            relations_section_list.append(
                f"({e["src_id"]},{e["relation"]},{e["tgt_id"]})"
            )

        relations_context = "\n".join(relations_section_list)
    else:
        relations_section_list = []
        relation_header = ",\t".join(
            [
                f"{enclose_string_with_quotes(data)}"
                for data in ["id", "source", "target", "relation"]
            ]
        )
        relations_section_list.append(relation_header)
        for i, e in enumerate(edges_data):
            raw_data = [i, e["src_tgt"][0], e["src_tgt"][1], e["relation"]]
            relations_section_list.append(
                ",\t".join([f"{enclose_string_with_quotes(data)}" for data in raw_data])
            )

        truncated_relations_list = truncate_list_by_token_size(
            relations_section_list,
            max_token_size=int(
                query_param.local_context_length * query_param.token_ratio_for_edge
            ),
            token_encoder=token_encoder,
        )
        if len(truncated_relations_list) == 0:
            relations_context = "No relations."
        else:
            relations_context = list_to_csv(truncated_relations_list)
    print(f"Form relation context time: {time.perf_counter() - tic:.2f}s")

    # build reasoning path context
    tic = time.perf_counter()
    reasoning_paths = []
    for i, e_list in enumerate(reasoning_path):
        reasoning_paths.append(
            f"{i}, " + "->".join([f"{enclose_string_with_quotes(e)}" for e in e_list])
        )
    truncated_reasoning_paths = truncate_list_by_token_size(
        reasoning_paths,
        max_token_size=int(
            query_param.local_context_length
            * query_param.token_ratio_for_reasoning_path
        ),
        token_encoder=token_encoder,
    )
    if len(truncated_reasoning_paths) == 0:
        reasoning_path_context = "No reasoning paths."
    else:
        reasoning_path_context = "\n".join(truncated_reasoning_paths)
    print(f"Form reasoning path time: {time.perf_counter() - tic:.2f}s")

    # build auxiliary data context
    tic = time.perf_counter()
    auxdata_context = "No auxiliary data."
    if len(aux_data) > 0:
        aux_keys = aux_data[0].keys()
        auxdata_header = ",\t".join(
            [f"{enclose_string_with_quotes(data)}" for data in aux_keys]
        )
        auxdata_section_list = [auxdata_header]
        for i, n in enumerate(aux_data):
            raw_data = [n.get(k, "UNKNOWN") for k in aux_keys]
            auxdata_section_list.append(
                ",\t".join([f"{enclose_string_with_quotes(data)}" for data in raw_data])
            )
        truncated_auxdata_list = truncate_list_by_token_size(
            auxdata_section_list,
            max_token_size=int(
                query_param.local_context_length
                * query_param.token_ratio_for_auxiliary_data
            ),
            token_encoder=token_encoder,
        )
        auxdata_context = list_to_csv(truncated_auxdata_list)
    print(f"Build auxiliary context time: {time.perf_counter() - tic:.2f}s")

    if query_param.traversal_type == "subgraphrag":
        return SUBGRAPHRAG_CONTEXT.format(relations_context=relations_context)
    else:
        return ALL_CONTEXT.format(
            cypher_query=cypher_query,
            entities_context=entities_context,
            relations_context=relations_context,
            reasoning_path_context=reasoning_path_context,
            auxdata_context=auxdata_context,
        )


async def gen_model_response(
    query: str, context: str, query_param: QueryParam, global_config: dict
) -> Tuple[str, int]:
    query = "Question:\n" + query
    if query[-1] != "?":
        query += "?"

    tic = time.perf_counter()
    query = "\n\n".join([context, query])
    conversation = [("system", icl_sys_prompt)]
    conversation.append(("user", icl_user_prompt))
    conversation.append(("assistant", icl_ass_prompt))
    sys_prompt = icl_sys_prompt + icl_user_prompt + icl_ass_prompt
    print(f"Form prompt time: {time.perf_counter() - tic:.2f}s")

    context_token_len = num_tokens(sys_prompt + query, global_config["token_encoder"])
    print(f"Context length: {context_token_len}")
    if context_token_len >= global_config["model_max_token_size"]:
        logger.error(
            f"Context length {context_token_len} exceeds the limit {global_config['model_max_token_size']}"
        )
        return PROMPTS["token_limit_exceeded"], 0

    tic = time.perf_counter()
    response = await global_config["model_func"](query, history_messages=conversation)
    print(f"LLM generate time: {time.perf_counter() - tic:.2f}s")

    if (
        "ans:" not in response.lower()
        or "ans: not available" in response.lower()
        or "ans: no information available" in response.lower()
    ):
        conversation.append(("user", query))

        tic = time.perf_counter()
        response = await global_config["model_func"](
            icl_cot_prompt, history_messages=conversation
        )
        print(f"LLM generate time: {time.perf_counter() - tic:.2f}s")

    return response, context_token_len


async def retrieve_and_generate(
    query: str,
    id_mapping: dict[str, ID],
    kg_inst: BaseGraphStorage,
    retriever: Callable,
    query_param: QueryParam,
    global_config: dict,
) -> tuple[str, int, int, List]:
    token_consumption = 0
    all_llm_calls = 0

    tic = time.perf_counter()
    try:
        retrieval_result: RetrievalResult = await retriever(
            query, id_mapping, kg_inst, query_param, global_config
        )
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return PROMPTS["fail_response"], token_consumption, all_llm_calls, []
    ndata = retrieval_result.nodes_data
    edata = retrieval_result.edges_data
    for data in ndata:
        id_mapping[data["name"]] = data["id"]
    token_consumption += retrieval_result.used_tokens
    all_llm_calls += retrieval_result.num_llm_calls
    logger.info(f"Get relations time: {time.perf_counter() - tic:.2f}s")
    logger.info(f"Using {len(ndata)} entites, {len(edata)} relations")

    tic = time.perf_counter()
    if query_param.traversal_type == "subgraphrag":
        sorted_ndata, sorted_edata = ndata, edata
    else:
        sorted_ndata, sorted_edata = await sort_entity_relation(ndata, edata, kg_inst)
    logger.info(f"Sort entities and relations time: {time.perf_counter() - tic:.2f}s")

    tic = time.perf_counter()
    context = form_entity_relation_context(
        retrieval_result.cypher_query,
        sorted_ndata,
        sorted_edata,
        retrieval_result.reasoning_paths,
        retrieval_result.auxiliary_data,
        query_param,
        global_config["token_encoder"],
    )
    logger.info(f"Form context time: {time.perf_counter() - tic:.2f}s")

    tic = time.perf_counter()
    response, context_token_len = await gen_model_response(
        query, context, query_param, global_config
    )
    logger.info(f"Generate response time: {time.perf_counter() - tic:.2f}s")
    token_consumption += context_token_len
    all_llm_calls += 1

    return response, token_consumption, all_llm_calls, retrieval_result.answer_list
