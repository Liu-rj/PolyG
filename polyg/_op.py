import re
import json
import asyncio
import tiktoken
import time
from typing import Union, List, Tuple, Dict
from collections import Counter, defaultdict
from ._splitter import SeparatorSplitter
from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    num_tokens,
    always_get_an_event_loop,
    enclose_string_with_quotes,
    list_to_csv,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
    CommunitySchema,
    TextChunkSchema,
    QueryParam,
)
from .prompt import (
    GRAPH_FIELD_SEP,
    PROMPTS,
    PHYSICS_GRAPH_SCHEMA,
    GOODREADS_GRAPH_SCHEMA,
    AMAZON_GRAPH_SCHEMA,
    FREEBASE_GRAPH_SCHEMA,
)


ALL_CONTEXT = """
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


def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def chunking_by_seperators(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):
    splitter = SeparatorSplitter(
        separators=[
            tiktoken_model.encode(s) for s in PROMPTS["default_text_separator"]
        ],
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_token]

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def get_chunks(new_docs, chunk_func=chunking_by_token_size, **chunk_func_params):
    inserting_chunks = {}

    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]

    ENCODER = tiktoken.encoding_for_model("gpt-4o")
    tokens = ENCODER.encode_batch(docs, num_threads=16)
    chunks = chunk_func(
        tokens, doc_keys=doc_keys, tiktoken_model=ENCODER, **chunk_func_params
    )

    for chunk in chunks:
        inserting_chunks.update(
            {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
        )

    return inserting_chunks


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["cheap_model_func"]
    llm_max_tokens = global_config["cheap_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knwoledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order
        ),
    )


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed * 100 // len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # it's undirected graph
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    return knwoledge_graph_inst


def _pack_single_community_by_sub_communities(
    community: SingleCommunitySchema,
    max_token_size: int,
    already_reports: dict[str, CommunitySchema],
) -> tuple[str, int]:
    # TODO
    all_sub_communities = [
        already_reports[k] for k in community["sub_communities"] if k in already_reports
    ]
    all_sub_communities = sorted(
        all_sub_communities, key=lambda x: x["occurrence"], reverse=True
    )
    may_trun_all_sub_communities = truncate_list_by_token_size(
        all_sub_communities,
        key=lambda x: x["report_string"],
        max_token_size=max_token_size,
    )
    sub_fields = ["id", "report", "rating", "importance"]
    sub_communities_describe = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,
                c["report_string"],
                c["report_json"].get("rating", -1),
                c["occurrence"],
            ]
            for i, c in enumerate(may_trun_all_sub_communities)
        ]
    )
    already_nodes = []
    already_edges = []
    for c in may_trun_all_sub_communities:
        already_nodes.extend(c["nodes"])
        already_edges.extend([tuple(e) for e in c["edges"]])
    return (
        sub_communities_describe,
        len(encode_string_by_tiktoken(sub_communities_describe)),
        set(already_nodes),
        set(already_edges),
    )


async def _pack_single_community_describe(
    knwoledge_graph_inst: BaseGraphStorage,
    community: SingleCommunitySchema,
    max_token_size: int = 12000,
    already_reports: dict[str, CommunitySchema] = {},
    global_config: dict = {},
) -> str:
    nodes_in_order = sorted(community["nodes"])
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    nodes_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_node(n) for n in nodes_in_order]
    )
    edges_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
    )
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    nodes_list_data = [
        [
            i,
            node_name,
            node_data.get("entity_type", "UNKNOWN"),
            node_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.node_degree(node_name),
        ]
        for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
    ]
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )
    edges_list_data = [
        [
            i,
            edge_name[0],
            edge_name[1],
            edge_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.edge_degree(*edge_name),
        ]
        for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
    ]
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

    # If context is exceed the limit and have sub-communities:
    report_describe = ""
    need_to_use_sub_communities = (
        truncated and len(community["sub_communities"]) and len(already_reports)
    )
    force_to_use_sub_communities = global_config["addon_params"].get(
        "force_to_use_sub_communities", False
    )
    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(
            f"Community {community['title']} exceeds the limit or you set force_to_use_sub_communities to True, using its sub-communities"
        )
        report_describe, report_size, contain_nodes, contain_edges = (
            _pack_single_community_by_sub_communities(
                community, max_token_size, already_reports
            )
        )
        report_exclude_nodes_list_data = [
            n for n in nodes_list_data if n[1] not in contain_nodes
        ]
        report_include_nodes_list_data = [
            n for n in nodes_list_data if n[1] in contain_nodes
        ]
        report_exclude_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) not in contain_edges
        ]
        report_include_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) in contain_edges
        ]
        # if report size is bigger than max_token_size, nodes and edges are []
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_nodes_list_data + report_include_nodes_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
        edges_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_edges_list_data + report_include_edges_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
    return f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""


def _community_report_json_to_str(parsed_output: dict) -> str:
    """refer official graphrag: index/graph/extractors/community_reports"""
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    def finding_summary(finding: dict):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")

    def finding_explanation(finding: dict):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}"


async def generate_community_report(
    community_report_kv: BaseKVStorage[CommunitySchema],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func: callable = global_config["model_func"]
    use_string_json_convert_func: callable = global_config[
        "convert_response_to_json_func"
    ]

    community_report_prompt = PROMPTS["community_report"]

    communities_schema = await knwoledge_graph_inst.community_schema()
    community_keys, community_values = (
        list(communities_schema.keys()),
        list(communities_schema.values()),
    )
    already_processed = 0

    async def _form_single_community_report(
        community: SingleCommunitySchema, already_reports: dict[str, CommunitySchema]
    ):
        nonlocal already_processed
        describe = await _pack_single_community_describe(
            knwoledge_graph_inst,
            community,
            max_token_size=global_config["model_max_token_size"],
            already_reports=already_reports,
            global_config=global_config,
        )
        prompt = community_report_prompt.format(input_text=describe)
        response = await use_llm_func(prompt, **llm_extra_kwargs)

        data = use_string_json_convert_func(response)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} communities\r",
            end="",
            flush=True,
        )
        return data

    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(community_keys, community_values)
                if v["level"] == level
            ]
        )
        this_level_communities_reports = await asyncio.gather(
            *[
                _form_single_community_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                    this_level_community_keys,
                    this_level_communities_reports,
                    this_level_community_values,
                )
            }
        )
    print()  # clear the progress bar
    await community_report_kv.upsert(community_datas)


async def _find_most_related_community_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    community_reports: BaseKVStorage[CommunitySchema],
):
    related_communities = []
    for node_d in node_datas:
        if "clusters" not in node_d:
            continue
        related_communities.extend(json.loads(node_d["clusters"]))
    related_community_dup_keys = [
        str(dp["cluster"])
        for dp in related_communities
        if dp["level"] <= query_param.level
    ]
    related_community_keys_counts = dict(Counter(related_community_dup_keys))
    _related_community_datas = await asyncio.gather(
        *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
    )
    related_community_datas = {
        k: v
        for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
        if v is not None
    }
    related_community_keys = sorted(
        related_community_keys_counts.keys(),
        key=lambda k: (
            related_community_keys_counts[k],
            related_community_datas[k]["report_json"].get("rating", -1),
        ),
        reverse=True,
    )
    sorted_community_datas = [
        related_community_datas[k] for k in related_community_keys
    ]

    use_community_reports = truncate_list_by_token_size(
        sorted_community_datas,
        key=lambda x: x["report_string"],
        max_token_size=query_param.local_max_token_for_community_report,
    )
    if query_param.local_community_single_one:
        use_community_reports = use_community_reports[:1]
    return use_community_reports


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    kg_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[kg_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[kg_inst.get_node(e) for e in all_one_hop_nodes]
    )
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.local_max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    id_mapping: dict[str, str],
    query_param: QueryParam,
    kg_inst: BaseGraphStorage,
):
    all_nodes = set(id_mapping.values())
    entry_ids = list(id_mapping.values())

    tic = time.time()
    print(f"Number of nodes before traversal: {len(all_nodes)}")
    print(
        f"Traversal type: {query_param.traversal_type}, depth: {query_param.edge_depth}"
    )

    if query_param.traversal_type == "BFS":
        all_edges = set()
        for depth in range(query_param.edge_depth):
            related_edges = await asyncio.gather(
                *[kg_inst.get_node_edges(node_id) for node_id in all_nodes]
            )
            for this_edges in related_edges:
                all_edges.update([tuple(sorted(e)) for e in this_edges])
                all_nodes.update([e[1] for e in this_edges])
            print(f"Number of nodes: {len(all_nodes)}")
        all_edges = list(all_edges)
        all_nodes = list(all_nodes)
        num_edges = len(all_edges)

        print(f"Number of nodes retrieved: {len(all_nodes)}")
        print(f"Number of edges retrieved: {num_edges}")
    elif query_param.traversal_type == "all_shortest_paths":
        all_node_path = await kg_inst.all_shortest_paths(entry_ids[0], entry_ids[1])

        print(f"Number of paths retrieved: {len(all_node_path)}")
        print(f"Number of edges retrieved: {sum([len(p) - 1 for p in all_node_path])}")
    elif query_param.traversal_type == "shortest_path":
        node_path = await kg_inst.shortest_path(entry_ids[0], entry_ids[1])
        all_node_path = [node_path]

        print(f"Number of paths retrieved: {len(all_node_path)}")
        print(f"Number of edges retrieved: {len(node_path) - 1}")
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
        all_edges_name = [
            (src["name"], tgt["name"]) for src, tgt in zip(src_node_pack, tgt_node_pack)
        ]
        all_edges_pack = await asyncio.gather(
            *[kg_inst.get_edge(e[0], e[1]) for e in all_edges]
        )
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
    elif query_param.traversal_type in ["shortest_path", "all_shortest_paths"]:
        tic = time.time()
        all_edges_data = []
        for node_path in all_node_path:
            path_data = []
            for i in range(len(node_path) - 1):
                node_data = await kg_inst.get_node(node_path[i])
                edge_data = await kg_inst.get_edge(node_path[i], node_path[i + 1])
                path_data.extend([node_data["name"], edge_data["relation"]])
            node_data = await kg_inst.get_node(node_path[-1])
            path_data.append(node_data["name"])
            all_edges_data.append(path_data)
        print(f"Collect path data time: {time.time() - tic:.2f}s")
    else:
        raise ValueError(f"Unknown traversal type: {query_param.traversal_type}")
    return (node_datas, all_edges_data)


def form_node_edge_context(
    node_datas: List[Dict],
    relation_datas: List[Dict],
    query_param: QueryParam,
):
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
        max_token_size=query_param.local_context_length
        * query_param.local_token_ratio_for_node,
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
    elif query_param.traversal_type in ["shortest_path", "all_shortest_paths"]:
        for i, e_list in enumerate(relation_datas):
            relations_section_list.append(
                f"{i}, "
                + "->".join([f"{enclose_string_with_quotes(e)}" for e in e_list])
            )
    else:
        raise ValueError(f"Unknown traversal type: {query_param.traversal_type}")

    truncated_relations_list = truncate_list_by_token_size(
        relations_section_list,
        max_token_size=query_param.local_context_length
        * query_param.local_token_ratio_for_edge,
    )
    relations_context = list_to_csv(truncated_relations_list)

    print(
        f"Build relation context time: {time.time() - tic:.2f}s, context length: {num_tokens(relations_context)} tokens"
    )
    return entities_context, relations_context


async def _build_local_query_context(
    query,
    id_mapping,
    kg_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    tic = time.time()
    node_datas, use_relations = await _find_most_related_edges_from_entities(
        id_mapping, query_param, kg_inst
    )
    print(f"Get relations time: {time.time() - tic:.2f}s")
    logger.info(f"Using {len(node_datas)} entites, {len(use_relations)} relations")

    entities_context, relations_context = form_node_edge_context(
        node_datas, use_relations, query_param
    )

    return ALL_CONTEXT.format(
        entities_context=entities_context,
        relations_context=relations_context,
        reasoning_path_context="",
    )


async def local_query(
    query,
    id_mapping,
    kg_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> tuple[str, int, int, str]:
    use_model_func = global_config["model_func"]
    try:
        tic = time.time()
        context = await _build_local_query_context(
            query,
            id_mapping,
            kg_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
            global_config,
        )
        print(f"Build context time: {time.time() - tic:.2f}s")
    except Exception as e:
        logger.error(f"Error in building local query context: {e}")
        return PROMPTS["fail_response"], 0, 0, "N/A"

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
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
        return PROMPTS["token_limit_exceeded"], 0, 0, "N/A"

    tic = time.time()
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    print(f"LLM generate time: {time.time() - tic:.2f}s")
    return response, form_response_tokens, 1, "N/A"


async def build_cypher_only_context(
    all_edges: List[Tuple],
    kg_inst: BaseGraphStorage,
    query_param: QueryParam,
):
    tic = time.time()
    # turn edges to a set of unique node ids
    entry_ids = set()
    for src, tgt in all_edges:
        entry_ids.add(src)
        entry_ids.add(tgt)
    node_datas = await asyncio.gather(*[kg_inst.get_node(nid) for nid in entry_ids])
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    print(f"Get node data time: {time.time() - tic:.2f}s")

    tic = time.time()
    src_node_pack = await asyncio.gather(*[kg_inst.get_node(e[0]) for e in all_edges])
    tgt_node_pack = await asyncio.gather(*[kg_inst.get_node(e[1]) for e in all_edges])
    all_edges_name = [
        (src["name"], tgt["name"]) for src, tgt in zip(src_node_pack, tgt_node_pack)
    ]
    all_edges_pack = await asyncio.gather(
        *[kg_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
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

    entities_context, relations_context = form_node_edge_context(
        node_datas, all_edges_data, query_param
    )

    return ALL_CONTEXT.format(
        entities_context=entities_context,
        relations_context=relations_context,
        reasoning_path_context="",
    )


async def cypher_only(
    query,
    id_mapping,
    kg_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> tuple[str, int, int, str]:
    use_model_func = global_config["model_func"]

    sys_prompt = PROMPTS["cypher_only_query"]
    if "Physics" in global_config["working_dir"]:
        sys_prompt = sys_prompt.format(graph_schema=PHYSICS_GRAPH_SCHEMA)
    elif "amazon" in global_config["working_dir"]:
        sys_prompt = sys_prompt.format(graph_schema=AMAZON_GRAPH_SCHEMA)
    elif "goodreads" in global_config["working_dir"]:
        sys_prompt = sys_prompt.format(graph_schema=GOODREADS_GRAPH_SCHEMA)
    elif (
        "webqsp" in global_config["working_dir"]
        or "cwq" in global_config["working_dir"]
    ):
        graph_schema = FREEBASE_GRAPH_SCHEMA.format(
            schema=global_config["graph_schema"]
        )
        sys_prompt = sys_prompt.format(graph_schema=graph_schema)
    else:
        raise NotImplementedError

    token_len = 0
    history_msgs = []
    response = None

    for i in range(query_param.failure_retries + 1):
        try:
            tic = time.time()
            prompt = f"query: {query}, id mapping: {id_mapping}"
            cur_token_len = num_tokens(sys_prompt + prompt) + sum(
                [num_tokens(m[1]) for m in history_msgs]
            )
            token_len += cur_token_len
            response = await use_model_func(
                prompt=prompt, system_prompt=sys_prompt, history_messages=history_msgs
            )
            cypher_query = response.split("```")[1].strip("cypher")
            print(f"Cypher query generation time: {time.time() - tic:.2f}s")
            print(f"Token length: {cur_token_len}")
            print("Generated cypher query:", cypher_query)

            tic = time.time()
            results = await kg_inst.exec_query(cypher_query)
            if results is None:
                return PROMPTS["fail_response"], token_len, 1, "N/A"
            all_edges = [(r["source"], r["target"]) for r in results]
            print(f"Query execution time: {time.time() - tic:.2f}s")

            break
        except Exception as e:
            if i == query_param.failure_retries:
                return PROMPTS["fail_response"], token_len, 1, "N/A"
            logger.error(f"Error: {e}")
            history_msgs.extend(
                [
                    ("user", prompt),
                    ("assistant", response),
                    ("user", PROMPTS["error_retry"].format(str(e))),
                ]
            )

    try:
        tic = time.time()
        context = await build_cypher_only_context(all_edges, kg_inst, query_param)
        print(f"Build context time: {time.time() - tic:.2f}s")
    except Exception as e:
        print(f"Error: {e}")
        return PROMPTS["fail_response"], token_len, 1, "N/A"

    if context is None:
        return PROMPTS["fail_response"], token_len, 1, "N/A"
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
        return PROMPTS["token_limit_exceeded"], token_len, 1, "N/A"

    tic = time.time()
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    print(f"LLM generate time: {time.time() - tic:.2f}s")
    return response, token_len + form_reponse_tokens, 2, "N/A"


async def guided_walk(
    query,
    id_mapping,
    kg_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> tuple[str, int, int, str]:
    use_model_func = global_config["model_func"]

    tic = time.time()
    sys_prompt = PROMPTS["cypher_query_prompt"]
    if "Physics" in global_config["working_dir"]:
        sys_prompt = sys_prompt.format(graph_schema=PHYSICS_GRAPH_SCHEMA)
    elif "amazon" in global_config["working_dir"]:
        sys_prompt = sys_prompt.format(graph_schema=AMAZON_GRAPH_SCHEMA)
    elif "goodreads" in global_config["working_dir"]:
        sys_prompt = sys_prompt.format(graph_schema=GOODREADS_GRAPH_SCHEMA)
    elif (
        "webqsp" in global_config["working_dir"]
        or "cwq" in global_config["working_dir"]
    ):
        graph_schema = FREEBASE_GRAPH_SCHEMA.format(
            schema=global_config["graph_schema"]
        )
        sys_prompt = sys_prompt.format(graph_schema=graph_schema)
    else:
        raise NotImplementedError

    token_len = 0
    history_msgs = []
    response = None

    for i in range(query_param.failure_retries + 1):
        try:
            prompt = f"query: {query}, id mapping: {id_mapping}"
            cur_token_len = num_tokens(sys_prompt + prompt) + sum(
                [num_tokens(m[1]) for m in history_msgs]
            )
            token_len += cur_token_len
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
            p_context, nodes, dests = await kg_inst.exec_query_and_get_path(
                cypher_query
            )
            if p_context is None:
                return PROMPTS["fail_response"], token_len, 1, "N/A"
            ret_names = set([d["name"] for d in dests])
            for node in nodes:
                id_mapping[node["name"]] = node["id"]
            print(f"Query execution time: {time.time() - tic:.2f}s")

            break
        except Exception as e:
            if i == query_param.failure_retries:
                return PROMPTS["fail_response"], token_len, 1, "N/A"
            logger.error(f"Error: {e}")
            history_msgs.extend(
                [
                    ("user", prompt),
                    ("assistant", response),
                    ("user", PROMPTS["error_retry"].format(str(e))),
                ]
            )

    related_edges = []
    if (
        "webqsp" in global_config["working_dir"]
        or "cwq" in global_config["working_dir"]
    ):
        rets = await asyncio.gather(*[kg_inst.get_node_edges(d["id"]) for d in dests])
        for edge_list in rets:
            related_edges.extend(edge_list)
        nodes.extend(
            await asyncio.gather(*[kg_inst.get_node(e[1]) for e in related_edges])
        )

    try:
        keys = ["id", "name", "node_type", "description"]
        entity_header = ",\t".join(
            [f"{enclose_string_with_quotes(data)}" for data in keys]
        )
        entites_section_list = [entity_header]
        for i, n in enumerate(nodes):
            raw_data = [n.get(k, "UNKNOWN") for k in keys]
            entites_section_list.append(
                ",\t".join([f"{enclose_string_with_quotes(data)}" for data in raw_data])
            )
        entities_context = list_to_csv(entites_section_list)

        relation_header = ",\t".join(
            [
                f"{enclose_string_with_quotes(data)}"
                for data in ["id", "source", "target", "relation"]
            ]
        )
        relations_section_list = [relation_header]
        for i, e in enumerate(related_edges):
            raw_data = [i, e[0], e[1], e[2]]
            relations_section_list.append(
                ",\t".join([f"{enclose_string_with_quotes(data)}" for data in raw_data])
            )
        relations_context = list_to_csv(relations_section_list)
    except Exception as e:
        print(f"Error: {e}")
        return PROMPTS["fail_response"], token_len, 1, "N/A"

    all_context = ALL_CONTEXT.format(
        entities_context=entities_context,
        relations_context=relations_context,
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
        return PROMPTS["token_limit_exceeded"], token_len, 1, "N/A"

    tic = time.time()
    final_response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    print(f"LLM generate time: {time.time() - tic:.2f}s")
    return final_response, token_len + form_reponse_tokens, 2, ", ".join(ret_names)


async def topk_csp(
    query,
    id_mapping,
    kg_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> tuple[str, int, int, str]:
    use_model_func = global_config["model_func"]

    tic = time.time()
    sys_prompt = PROMPTS["cypher_path_search_prompt"]
    if "Physics" in global_config["working_dir"]:
        sys_prompt = sys_prompt.format(graph_schema=PHYSICS_GRAPH_SCHEMA)
    elif "amazon" in global_config["working_dir"]:
        sys_prompt = sys_prompt.format(graph_schema=AMAZON_GRAPH_SCHEMA)
    elif "goodreads" in global_config["working_dir"]:
        sys_prompt = sys_prompt.format(graph_schema=GOODREADS_GRAPH_SCHEMA)
    elif (
        "webqsp" in global_config["working_dir"]
        or "cwq" in global_config["working_dir"]
    ):
        graph_schema = FREEBASE_GRAPH_SCHEMA.format(
            schema=global_config["graph_schema"]
        )
        sys_prompt = sys_prompt.format(graph_schema=graph_schema)
    else:
        raise NotImplementedError

    token_len = 0
    history_msgs = []
    response = None

    for i in range(query_param.failure_retries + 1):
        try:
            prompt = f"query: {query}, id mapping: {id_mapping}"
            cur_token_len = num_tokens(sys_prompt + prompt) + sum(
                [num_tokens(m[1]) for m in history_msgs]
            )
            token_len += cur_token_len
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
            p_context, nodes, _ = await kg_inst.exec_query_and_get_path(cypher_query)
            if p_context is None:
                return PROMPTS["fail_response"], token_len, 1, "N/A"
            for node in nodes:
                id_mapping[node["name"]] = node["id"]
            print(f"Query execution time: {time.time() - tic:.2f}s")

            break
        except Exception as e:
            if i == query_param.failure_retries:
                return PROMPTS["fail_response"], token_len, 1, "N/A"
            logger.error(f"Error: {e}")
            history_msgs.extend(
                [
                    ("user", prompt),
                    ("assistant", response),
                    ("user", PROMPTS["error_retry"].format(str(e))),
                ]
            )

    try:
        keys = ["name", "node_type", "description"]
        entity_header = ",\t".join(
            [f"{enclose_string_with_quotes(data)}" for data in keys]
        )
        entites_section_list = [entity_header]
        for i, n in enumerate(nodes):
            raw_data = [n.get(k, "UNKNOWN") for k in keys]
            entites_section_list.append(
                ",\t".join([f"{enclose_string_with_quotes(data)}" for data in raw_data])
            )
        entities_context = list_to_csv(entites_section_list)
    except Exception as e:
        print(f"Error: {e}")
        return PROMPTS["fail_response"], token_len, 1, "N/A"

    all_context = ALL_CONTEXT.format(
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
        return PROMPTS["token_limit_exceeded"], token_len, 1, "N/A"

    tic = time.time()
    final_response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    print(f"LLM generate time: {time.time() - tic:.2f}s")
    return final_response, token_len + form_reponse_tokens, 2, "N/A"


def batch_local_query(
    query: List[str],
    kg_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> List[str]:
    use_model_func = global_config["model_func"]
    tic = time.time()
    loop = always_get_an_event_loop()
    batch_context = loop.run_until_complete(
        asyncio.gather(
            *[
                _build_local_query_context(
                    q,
                    kg_inst,
                    entities_vdb,
                    community_reports,
                    text_chunks_db,
                    query_param,
                    global_config,
                )
                for q in query
            ]
        )
    )
    print(f"Build context time: {time.time() - tic:.2f}s")
    for c in batch_context:
        assert (
            num_tokens(c) < global_config["model_max_token_size"]
        ), f"context length: {num_tokens(c)}"
        print(f"context: {c}")
        print(f"Context length: {num_tokens(c)} tokens")

    if query_param.only_need_context:
        return batch_context
    for i, context in enumerate(batch_context):
        if context is None:
            batch_context[i] = PROMPTS["fail_response"]

    tic = time.time()
    batch_sys_prompt = []
    for context in batch_context:
        sys_prompt_temp = PROMPTS["local_rag_response"]
        sys_prompt = sys_prompt_temp.format(
            context_data=context, response_type=query_param.response_type
        )
        batch_sys_prompt.append(sys_prompt)
    print(f"Form prompt time: {time.time() - tic:.2f}s")

    tic = time.time()
    loop = always_get_an_event_loop()
    response = loop.run_until_complete(
        use_model_func(
            query,
            system_prompt=batch_sys_prompt,
        )
    )
    print(f"LLM generate time: {time.time() - tic:.2f}s")
    return response


async def _map_global_communities(
    query: str,
    communities_data: list[CommunitySchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_string_json_convert_func = global_config["convert_response_to_json_func"]
    use_model_func = global_config["model_func"]
    community_groups = []
    while len(communities_data):
        this_group = truncate_list_by_token_size(
            communities_data,
            key=lambda x: x["report_string"],
            max_token_size=query_param.global_max_token_for_community_report,
        )
        community_groups.append(this_group)
        communities_data = communities_data[len(this_group) :]

    async def _process(community_truncated_datas: list[CommunitySchema]) -> dict:
        communities_section_list = [["id", "content", "rating", "importance"]]
        for i, c in enumerate(community_truncated_datas):
            communities_section_list.append(
                [
                    i,
                    c["report_string"],
                    c["report_json"].get("rating", 0),
                    c["occurrence"],
                ]
            )
        community_context = list_of_list_to_csv(communities_section_list)
        sys_prompt_temp = PROMPTS["global_map_rag_points"]
        sys_prompt = sys_prompt_temp.format(context_data=community_context)
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
            **query_param.global_special_community_map_llm_kwargs,
        )
        data = use_string_json_convert_func(response)
        return data.get("points", [])

    logger.info(f"Grouping to {len(community_groups)} groups for global search")
    responses = await asyncio.gather(*[_process(c) for c in community_groups])
    return responses


async def global_query(
    query,
    kg_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    community_schema = await kg_inst.community_schema()
    community_schema = {
        k: v for k, v in community_schema.items() if v["level"] <= query_param.level
    }
    if not len(community_schema):
        return PROMPTS["fail_response"]
    use_model_func = global_config["model_func"]

    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )
    sorted_community_schemas = sorted_community_schemas[
        : query_param.global_max_consider_community
    ]
    community_datas = await community_reports.get_by_ids(
        [k[0] for k in sorted_community_schemas]
    )
    community_datas = [c for c in community_datas if c is not None]
    community_datas = [
        c
        for c in community_datas
        if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating
    ]
    community_datas = sorted(
        community_datas,
        key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
        reverse=True,
    )
    logger.info(f"Revtrieved {len(community_datas)} communities")

    map_communities_points = await _map_global_communities(
        query, community_datas, query_param, global_config
    )
    final_support_points = []
    for i, mc in enumerate(map_communities_points):
        for point in mc:
            if "description" not in point:
                continue
            final_support_points.append(
                {
                    "analyst": i,
                    "answer": point["description"],
                    "score": point.get("score", 1),
                }
            )
    final_support_points = [p for p in final_support_points if p["score"] > 0]
    if not len(final_support_points):
        return PROMPTS["fail_response"]
    final_support_points = sorted(
        final_support_points, key=lambda x: x["score"], reverse=True
    )
    final_support_points = truncate_list_by_token_size(
        final_support_points,
        key=lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report,
    )
    points_context = []
    for dp in final_support_points:
        points_context.append(
            f"""----Analyst {dp["analyst"]}----
Importance Score: {dp["score"]}
{dp["answer"]}
"""
        )
    points_context = "\n".join(points_context)
    if query_param.only_need_context:
        return points_context
    sys_prompt_temp = PROMPTS["global_reduce_rag_response"]
    response = await use_model_func(
        query,
        sys_prompt_temp.format(
            report_data=points_context, response_type=query_param.response_type
        ),
    )
    return response


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_model_func = global_config["model_func"]
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.naive_max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    if query_param.only_need_context:
        return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response
