import asyncio
import time
from typing import Union, List, Tuple, Dict
from .utils import (
    logger,
    truncate_list_by_token_size,
    num_tokens,
    enclose_string_with_quotes,
    list_to_csv,
)
from .base import BaseGraphStorage, QueryParam, ID
from .prompt import (
    PROMPTS,
    SCHEMA_MAP,
)


ALL_CONTEXT = """
-----Cypher Query-----
```cypher
{cypher_query}
```

-----Entities-----
```csv
{entities_context}
```

-----Relationships-----
```csv
{relations_context}
```

-----Reasoning Path-----
{reasoning_path_context}
"""


async def _find_most_related_edges_from_entities(
    id_mapping: dict[str, str],
    query_param: QueryParam,
    kg_inst: BaseGraphStorage,
) -> Tuple[List[Dict], List[Dict], str]:
    all_nodes = set(id_mapping.values())
    entry_ids = list(id_mapping.values())

    tic = time.time()
    print(f"Number of nodes before traversal: {len(all_nodes)}")
    print(
        f"Traversal type: {query_param.traversal_type}, depth: {query_param.edge_depth}"
    )

    all_edges = set()
    all_node_path = []
    if query_param.traversal_type == "BFS":
        for _ in range(query_param.edge_depth):
            related_edges = await asyncio.gather(
                *[kg_inst.get_node_edges(node_id) for node_id in all_nodes]
            )
            for this_edges in related_edges:
                all_edges.update([tuple(sorted(e[0:2])) for e in this_edges])
                all_nodes.update([e[1] for e in this_edges])
            print(f"Number of nodes: {len(all_nodes)}")
        all_edges = list(all_edges)
        all_nodes = list(all_nodes)
        num_edges = len(all_edges)

        print(f"Number of nodes retrieved: {len(all_nodes)}")
        print(f"Number of edges retrieved: {num_edges}")

        cypher_query = (
            "MATCH (s:self.namespace)-[r]-(t:self.namespace) WHERE s.id = $source_id\n"
            "RETURN s.id AS source, t.id AS target, Type(r) AS relation"
        )
    elif query_param.traversal_type == "topk_shortest_paths":
        all_node_path = await kg_inst.topk_shortest_paths(entry_ids[0], entry_ids[1])

        print(f"Number of paths retrieved: {len(all_node_path)}")
        print(f"Number of edges retrieved: {sum([len(p) - 1 for p in all_node_path])}")

        cypher_query = (
            "MATCH p = SHORTEST 20 (s:self.namespace {id: $source_id})-[*]->"
            "(t:self.namespace {id: $target_id})\n"
            "RETURN [n in nodes(p) | n.id] AS path"
        )
    else:
        raise ValueError(f"Unknown traversal type: {query_param.traversal_type}")

    print(f"Traversal time: {time.time() - tic:.2f}s")

    tic = time.time()
    node_datas = await asyncio.gather(*[kg_inst.get_node(nid) for nid in all_nodes])
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[kg_inst.node_degree(nid) for nid in all_nodes]
    )
    node_datas = [
        {**n, "rank": d} for n, d in zip(node_datas, node_degrees) if n is not None
    ]
    node_datas = sorted(node_datas, key=lambda x: x["rank"], reverse=True)
    for it, node_data in enumerate(node_datas):
        id_mapping[node_data["name"]] = node_data["id"]
    print(f"Get node data time: {time.time() - tic:.2f}s")

    if query_param.traversal_type == "BFS":
        tic = time.time()
        src_node_pack = await asyncio.gather(
            *[kg_inst.get_node(e[0]) for e in all_edges]
        )
        tgt_node_pack = await asyncio.gather(
            *[kg_inst.get_node(e[1]) for e in all_edges]
        )
        # assert all are not None
        assert all(x is not None for x in src_node_pack)
        assert all(x is not None for x in tgt_node_pack)
        all_edges_name = [
            (src["name"], tgt["name"]) for src, tgt in zip(src_node_pack, tgt_node_pack)  # type: ignore
        ]
        all_edges_pack = await asyncio.gather(
            *[kg_inst.get_edge(e[0], e[1]) for e in all_edges]
        )
        assert all(x is not None for x in all_edges_pack)
        print(f"Get edge data time: {time.time() - tic:.2f}s")

        tic = time.time()
        all_edges_degree = await asyncio.gather(
            *[kg_inst.edge_degree(e[0], e[1]) for e in all_edges]
        )
        print(f"Get edge degree time: {time.time() - tic:.2f}s")

        tic = time.time()
        all_edges_data = [
            {"src_tgt": k, "rank": r, **v}
            for k, r, v in zip(all_edges_name, all_edges_degree, all_edges_pack)
            if v is not None
        ]
        print(f"Combine edge data time: {time.time() - tic:.2f}s")

        tic = time.time()
        all_edges_data = sorted(all_edges_data, key=lambda x: x["rank"], reverse=True)
        print(f"Sort edge data time: {time.time() - tic:.2f}s")

        # all_edges_data = truncate_list_by_token_size(
        #     all_edges_data,
        #     key=lambda x: x["relation"],
        #     max_token_size=query_param.local_max_token_for_local_context,
        # )
        # print(f"Number of edges after truncation: {len(all_edges_data)}")
    elif query_param.traversal_type == "topk_shortest_paths":
        tic = time.time()
        all_edges_data = []
        for node_path in all_node_path:
            path_data = []
            for i in range(len(node_path) - 1):
                node_data = await kg_inst.get_node(node_path[i])
                edge_data = await kg_inst.get_edge(node_path[i], node_path[i + 1])
                assert node_data is not None and edge_data is not None
                path_data.extend([node_data["name"], edge_data["relation"]])
            node_data = await kg_inst.get_node(node_path[-1])
            assert node_data is not None
            path_data.append(node_data["name"])
            all_edges_data.append(path_data)
        print(f"Collect path data time: {time.time() - tic:.2f}s")
    else:
        raise ValueError(f"Unknown traversal type: {query_param.traversal_type}")
    return (node_datas, all_edges_data, cypher_query)


def form_node_edge_context(
    node_datas: List[Dict],
    relation_datas: List[Dict],
    query_param: QueryParam,
) -> Tuple[str, str]:
    tic = time.time()
    keys = ["name", "node_type", "description"]
    entity_header = ",\t".join([f"{enclose_string_with_quotes(data)}" for data in keys])
    entites_section_list = [entity_header]
    for i, n in enumerate(node_datas):
        raw_data = [n.get(k, "UNKNOWN") for k in keys]
        entites_section_list.append(
            ",\t".join([f"{enclose_string_with_quotes(data)}" for data in raw_data])
        )
    truncated_entities_list = truncate_list_by_token_size(
        entites_section_list,
        max_token_size=int(
            query_param.local_context_length * query_param.local_token_ratio_for_node
        ),
    )
    entities_context = list_to_csv(truncated_entities_list)
    print(
        f"Build entity context time: {time.time() - tic:.2f}s, context length: {num_tokens(entities_context)} tokens"
    )

    tic = time.time()
    relations_section_list = []
    if query_param.traversal_type in ["BFS", "cypher_only"]:
        relation_header = ",\t".join(
            [
                f"{enclose_string_with_quotes(data)}"
                for data in ["id", "source", "target", "relation"]
            ]
        )
        relations_section_list.append(relation_header)
        for i, e in enumerate(relation_datas):
            raw_data = [i, e["src_tgt"][0], e["src_tgt"][1], e["relation"]]
            relations_section_list.append(
                ",\t".join([f"{enclose_string_with_quotes(data)}" for data in raw_data])
            )
    elif query_param.traversal_type == "topk_shortest_paths":
        for i, e_list in enumerate(relation_datas):
            relations_section_list.append(
                f"{i}, "
                + "->".join([f"{enclose_string_with_quotes(e)}" for e in e_list])
            )
    else:
        raise ValueError(f"Unknown traversal type: {query_param.traversal_type}")

    truncated_relations_list = truncate_list_by_token_size(
        relations_section_list,
        max_token_size=int(
            query_param.local_context_length * query_param.local_token_ratio_for_edge
        ),
    )
    relations_context = list_to_csv(truncated_relations_list)

    print(
        f"Build relation context time: {time.time() - tic:.2f}s, context length: {num_tokens(relations_context)} tokens"
    )
    return entities_context, relations_context


async def _build_local_query_context(
    query: str,
    id_mapping: dict[str, str],
    kg_inst: BaseGraphStorage,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    tic = time.time()
    node_datas, use_relations, cypher_query = (
        await _find_most_related_edges_from_entities(id_mapping, query_param, kg_inst)
    )
    print(f"Get relations time: {time.time() - tic:.2f}s")
    logger.info(f"Using {len(node_datas)} entites, {len(use_relations)} relations")

    entities_context, relations_context = form_node_edge_context(
        node_datas, use_relations, query_param
    )

    return ALL_CONTEXT.format(
        cypher_query=cypher_query,
        entities_context=entities_context,
        relations_context=relations_context,
        reasoning_path_context="",
    )


async def local_query(
    query: str,
    id_mapping: dict[str, str],
    kg_inst: BaseGraphStorage,
    query_param: QueryParam,
    global_config: dict,
) -> tuple[str, int, int, List]:
    use_model_func = global_config["model_func"]
    try:
        tic = time.time()
        context = await _build_local_query_context(
            query,
            id_mapping,
            kg_inst,
            query_param,
            global_config,
        )
        print(f"Build context time: {time.time() - tic:.2f}s")
    except Exception as e:
        logger.error(f"Error in building local query context: {e}")
        return PROMPTS["fail_response"], 0, 0, []

    tic = time.time()
    sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    print(f"Form prompt time: {time.time() - tic:.2f}s")

    form_response_tokens = num_tokens(sys_prompt + query)
    print(f"Context length: {form_response_tokens}")
    if form_response_tokens >= global_config["model_max_token_size"]:
        logger.error(
            f"Context length {form_response_tokens} exceeds the limit {global_config['model_max_token_size']}"
        )
        return PROMPTS["token_limit_exceeded"], 0, 0, []

    tic = time.time()
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    print(f"LLM generate time: {time.time() - tic:.2f}s")
    return response, form_response_tokens, 1, []


async def build_schema_example(kg_inst: BaseGraphStorage, node_ids: List[ID]):
    all_edges = set()
    related_edges = await asyncio.gather(
        *[kg_inst.get_node_edges(node_id) for node_id in node_ids]
    )
    for this_edges in related_edges:
        all_edges.update(this_edges)

    relation_header = ",\t".join(
        [
            f"{enclose_string_with_quotes(data)}"
            for data in ["id", "source", "target", "relation"]
        ]
    )
    relations_section_list = [relation_header]
    for i, e in enumerate(all_edges):
        raw_data = [i, e[0], e[1], e[2]]
        relations_section_list.append(
            ",\t".join([f"{enclose_string_with_quotes(data)}" for data in raw_data])
        )
    return list_to_csv(relations_section_list)


async def cypher_only(
    query: str,
    id_mapping: dict[str, ID],
    kg_inst: BaseGraphStorage,
    query_param: QueryParam,
    global_config: dict,
) -> tuple[str, int, int, List]:
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
            tic = time.time()
            cur_token_len = num_tokens(sys_prompt + prompt)
            token_len += cur_token_len
            llm_calls += 1
            response = await use_model_func(
                prompt=prompt,
                system_prompt=sys_prompt,
                history_messages=history_msgs,
            )
            cypher_query = response.split("```")[1].strip("cypher")
            print(f"Cypher query generation time: {time.time() - tic:.2f}s")
            print(f"Token length: {cur_token_len}")
            print("Generated cypher query:", cypher_query)

            tic = time.time()
            results = await kg_inst.exec_query(cypher_query)
            print(f"Query execution time: {time.time() - tic:.2f}s")

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

    if len(results) == 0:
        return PROMPTS["fail_response"], token_len, llm_calls, []

    tic = time.time()
    keys = results[0].keys()
    entity_header = ",\t".join([f"{enclose_string_with_quotes(data)}" for data in keys])
    entites_section_list = [entity_header]
    for i, n in enumerate(results):
        raw_data = [n.get(k, "UNKNOWN") for k in keys]
        entites_section_list.append(
            ",\t".join([f"{enclose_string_with_quotes(data)}" for data in raw_data])
        )
    truncated_entities_list = truncate_list_by_token_size(
        entites_section_list,
        max_token_size=int(
            query_param.local_context_length * query_param.local_token_ratio_for_node
        ),
    )
    entities_context = list_to_csv(truncated_entities_list)
    print(
        f"Build entity context time: {time.time() - tic:.2f}s, context length: {num_tokens(entities_context)} tokens"
    )

    context = ALL_CONTEXT.format(
        cypher_query=cypher_query,
        entities_context=entities_context,
        relations_context="",
        reasoning_path_context="",
    )

    tic = time.time()
    sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    print(f"Form prompt time: {time.time() - tic:.2f}s")

    form_reponse_tokens = num_tokens(sys_prompt + query)
    print(f"Token length: {form_reponse_tokens}")
    if form_reponse_tokens >= global_config["model_max_token_size"]:
        logger.error(
            f"Context length {form_reponse_tokens} exceeds the limit {global_config['model_max_token_size']}"
        )
        return PROMPTS["token_limit_exceeded"], token_len, llm_calls, []

    tic = time.time()
    response = await use_model_func(query, system_prompt=sys_prompt)
    llm_calls += 1
    print(f"LLM generate time: {time.time() - tic:.2f}s")

    return response, token_len + form_reponse_tokens, llm_calls, []


async def guided_walk(
    query: str,
    id_mapping: dict[str, ID],
    kg_inst: BaseGraphStorage,
    query_param: QueryParam,
    global_config: dict,
) -> tuple[str, int, int, List]:
    use_model_func = global_config["model_func"]

    tic = time.time()
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
            cur_token_len = num_tokens(sys_prompt + prompt)
            token_len += cur_token_len
            llm_calls += 1
            response = await use_model_func(
                prompt=prompt,
                system_prompt=sys_prompt,
                history_messages=history_msgs,
            )
            print(response)
            cypher_query = response.split("```")[1].strip("cypher")
            print(f"Cypher query generation time: {time.time() - tic:.2f}s")
            print(f"Token length: {cur_token_len}")
            print("Generated cypher query:", cypher_query)

            # tic = time.time()
            # p_context, node_ids, dest_ids = await kg_inst.exec_query_and_get_path(
            #     cypher_query
            # )
            # print(p_context)
            # print(node_ids)
            # print(dest_ids)
            # if len(p_context) == 0 and len(node_ids) == 0:
            #     raise ValueError("No result found, please adjust the query")
            # print(f"Query execution time: {time.time() - tic:.2f}s")

            tic = time.time()
            results = await kg_inst.exec_query(cypher_query)
            if len(results) == 0:
                raise ValueError("No result found, please adjust the query")
            ret_ids = [r["id"] for r in results]
            print(f"Query execution time: {time.time() - tic:.2f}s")

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

    # if len(p_context) == 0 and len(node_ids) == 0:
    #     return PROMPTS["fail_response"], token_len, llm_calls, []

    # dest_datas = await asyncio.gather(*[kg_inst.get_node(d) for d in dest_ids])
    # assert all(x is not None for x in dest_datas)
    # ret_names = set([d["name"] for d in dest_datas])  # type: ignore

    # related_edges = []
    # if global_config["dataset"] in ["webqsp", "cwq"]:
    #     rets = await asyncio.gather(*[kg_inst.get_node_edges(d) for d in dest_ids])
    #     for edge_list in rets:
    #         related_edges.extend(edge_list)
    #     node_ids.update([e[1] for e in related_edges])

    # node_datas = await asyncio.gather(*[kg_inst.get_node(nid) for nid in node_ids])
    # assert all(x is not None for x in node_datas)
    # for it, node in enumerate(node_datas):
    #     id_mapping[node["name"]] = node["id"]  # type: ignore

    # keys = ["id", "name", "node_type", "description"]
    # entity_header = ",\t".join([f"{enclose_string_with_quotes(data)}" for data in keys])
    # entites_section_list = [entity_header]
    # for i, n in enumerate(node_datas):
    #     assert n is not None
    #     raw_data = [n.get(k, "UNKNOWN") for k in keys]
    #     entites_section_list.append(
    #         ",\t".join([f"{enclose_string_with_quotes(data)}" for data in raw_data])
    #     )
    # entities_context = list_to_csv(entites_section_list)

    # relation_header = ",\t".join(
    #     [
    #         f"{enclose_string_with_quotes(data)}"
    #         for data in ["id", "source", "target", "relation"]
    #     ]
    # )
    # relations_section_list = [relation_header]
    # for i, e in enumerate(related_edges):
    #     raw_data = [i, e[0], e[1], e[2]]
    #     relations_section_list.append(
    #         ",\t".join([f"{enclose_string_with_quotes(data)}" for data in raw_data])
    #     )
    # relations_context = list_to_csv(relations_section_list)

    tic = time.time()
    entry_ids = list(id_mapping.values())
    ret_ids = entry_ids + ret_ids
    node_datas = await asyncio.gather(*[kg_inst.get_node(nid) for nid in ret_ids])
    node_datas = [n for n in node_datas if n is not None] # remove None
    for item in node_datas:
        id_mapping[item["name"]] = item["id"] # type: ignore
    ret_names = [n["name"] for n in node_datas[len(entry_ids) :]] # type: ignore
    print(ret_names)
    print(f"Get node data time: {time.time() - tic:.2f}s")

    keys = ["name", "node_type", "description"]
    entity_header = ",\t".join([f"{enclose_string_with_quotes(data)}" for data in keys])
    entites_section_list = [entity_header]
    for i, n in enumerate(node_datas):
        raw_data = [n.get(k, "UNKNOWN") for k in keys] # type: ignore
        entites_section_list.append(
            ",\t".join([f"{enclose_string_with_quotes(data)}" for data in raw_data])
        )
    entities_context = list_to_csv(entites_section_list)

    all_context = ALL_CONTEXT.format(
        cypher_query=cypher_query,
        entities_context=entities_context,
        relations_context="",
        reasoning_path_context="",
    )

    tic = time.time()
    if global_config["dataset"] in ["webqsp", "cwq"]:
        sys_prompt_temp = PROMPTS["guided_walk_response"]
    else:
        sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=all_context, response_type=query_param.response_type
    )
    print(f"Form prompt time: {time.time() - tic:.2f}s")

    form_reponse_tokens = num_tokens(sys_prompt + query)
    print(f"Token length: {form_reponse_tokens}")
    if form_reponse_tokens >= global_config["model_max_token_size"]:
        logger.error(
            f"Context length {form_reponse_tokens} exceeds the limit {global_config['model_max_token_size']}"
        )
        return PROMPTS["token_limit_exceeded"], token_len, llm_calls, []

    tic = time.time()
    response = await use_model_func(query, system_prompt=sys_prompt)
    llm_calls += 1
    print(f"LLM generate time: {time.time() - tic:.2f}s")

    return response, token_len + form_reponse_tokens, llm_calls, list(ret_names)


async def topk_csp(
    query: str,
    id_mapping: dict[str, ID],
    kg_inst: BaseGraphStorage,
    query_param: QueryParam,
    global_config: dict,
) -> tuple[str, int, int, List]:
    use_model_func = global_config["model_func"]

    tic = time.time()
    graph_schema = SCHEMA_MAP[global_config["dataset"]]
    if global_config["dataset"] in ["webqsp", "cwq"]:
        schema_example = await build_schema_example(kg_inst, list(id_mapping.values()))
        graph_schema = graph_schema.format(
            schema=global_config["concrete_graph_schema"],
            benchmark=global_config["dataset"],
            example=schema_example,
        )
    sys_prompt = PROMPTS["cypher_path_search_prompt"].format(graph_schema=graph_schema)

    token_len = 0
    history_msgs = []
    response = None
    llm_calls = 0

    p_context, node_ids = [], set()
    cypher_query = ""
    prompt = f"query: {query}, id mapping: {id_mapping}"
    for i in range(query_param.failure_retries + 1):
        try:
            cur_token_len = num_tokens(sys_prompt + prompt)
            token_len += cur_token_len
            llm_calls += 1
            response = await use_model_func(
                prompt=prompt,
                system_prompt=sys_prompt,
                history_messages=history_msgs,
            )
            cypher_query = response.split("```")[1].strip("cypher")
            print(f"Cypher query generation time: {time.time() - tic:.2f}s")
            print(f"Token length: {cur_token_len}")
            print("Generated cypher query:", cypher_query)

            tic = time.time()
            p_context, node_ids, _ = await kg_inst.exec_query_and_get_path(cypher_query)
            if len(p_context) == 0:
                raise ValueError("No path found, please adjust the query")
            print(f"Query execution time: {time.time() - tic:.2f}s")

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

    if len(p_context) == 0 and len(node_ids) == 0:
        return PROMPTS["fail_response"], token_len, llm_calls, []

    node_datas = await asyncio.gather(*[kg_inst.get_node(nid) for nid in node_ids])
    node_datas = [n for n in node_datas if n is not None] # remove None
    for node in node_datas:
        id_mapping[node["name"]] = node["id"]  # type: ignore

    keys = ["name", "node_type", "description"]
    entity_header = ",\t".join([f"{enclose_string_with_quotes(data)}" for data in keys])
    entities_section_list = [entity_header]
    for i, n in enumerate(node_datas):
        raw_data = [n.get(k, "UNKNOWN") for k in keys]
        entities_section_list.append(
            ",\t".join([f"{enclose_string_with_quotes(data)}" for data in raw_data])
        )
    entities_context = list_to_csv(entities_section_list)

    all_context = ALL_CONTEXT.format(
        cypher_query=cypher_query,
        entities_context=entities_context,
        relations_context="",
        reasoning_path_context=p_context,
    )

    tic = time.time()
    sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=all_context, response_type=query_param.response_type
    )
    print(f"Form prompt time: {time.time() - tic:.2f}s")

    form_reponse_tokens = num_tokens(sys_prompt + query)
    print(f"Token length: {form_reponse_tokens}")
    if form_reponse_tokens >= global_config["model_max_token_size"]:
        logger.error(
            f"Context length {form_reponse_tokens} exceeds the limit {global_config['model_max_token_size']}"
        )
        return PROMPTS["token_limit_exceeded"], token_len, llm_calls, []

    tic = time.time()
    response = await use_model_func(query, system_prompt=sys_prompt)
    llm_calls += 1
    print(f"LLM generate time: {time.time() - tic:.2f}s")

    return response, token_len + form_reponse_tokens, llm_calls, []
