import asyncio
import time
from typing import List, Tuple, Dict
from ..base import BaseGraphStorage, QueryParam, ID, RetrievalResult
from ..prompt import PROMPTS, SCHEMA_MAP
from ..utils import logger


async def shortest_path_retriever(
    query: str,
    id_mapping: dict[str, ID],
    kg_inst: BaseGraphStorage,
    query_param: QueryParam,
    global_config: dict,
) -> RetrievalResult:
    cypher_query = (
        "MATCH p = SHORTEST 20 (s:self.namespace {id: $source_id})-[*]-"
        "(t:self.namespace {id: $target_id})\n"
        "RETURN path"
    )

    tic = time.perf_counter()
    entry_ids = list(id_mapping.values())
    all_edge_paths = await kg_inst.topk_shortest_paths(entry_ids[0], entry_ids[1])
    print(f"Shortest path retrieval time: {time.perf_counter() - tic:.2f}s")
    print(f"Number of paths retrieved: {len(all_edge_paths)}")
    print(f"Number of edges retrieved: {sum([len(p) for p in all_edge_paths])}")

    tic = time.perf_counter()
    all_paths = []
    all_nodes = set()
    for edge_path in all_edge_paths:
        path_data = []
        for src_id, relation, tgt_id in edge_path:
            all_nodes.add(src_id)
            all_nodes.add(tgt_id)
            node_data = await kg_inst.get_node(src_id)
            assert node_data is not None, f"Node {src_id} not found"
            path_data.extend([node_data["name"], relation])
        # Add the last node's name
        if edge_path:
            last_node_data = await kg_inst.get_node(edge_path[-1][2])
            assert last_node_data is not None, f"Node {edge_path[-1][2]} not found"
            path_data.append(last_node_data["name"])
        all_paths.append(path_data)
    print(f"Collect path data time: {time.perf_counter() - tic:.2f}s")

    tic = time.perf_counter()
    nodes_data = await asyncio.gather(*[kg_inst.get_node(nid) for nid in all_nodes])
    assert all([n is not None for n in nodes_data])
    logger.info(f"Node fetching time: {time.perf_counter() - tic:.2f}s")

    return RetrievalResult(
        cypher_query=cypher_query,
        nodes_data=nodes_data,  # type: ignore
        edges_data=[],
        reasoning_paths=all_paths,
        auxiliary_data=[],
        used_tokens=0,
        num_llm_calls=0,
        answer_list=[],
    )
