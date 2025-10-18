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
        "MATCH p = SHORTEST 20 (s:self.namespace {id: $source_id})-[*]->"
        "(t:self.namespace {id: $target_id})\n"
        "RETURN [n in nodes(p) | n.id] AS path"
    )

    tic = time.perf_counter()
    entry_ids = list(id_mapping.values())
    all_node_path = await kg_inst.topk_shortest_paths(entry_ids[0], entry_ids[1])
    print(f"Shortest path retrieval time: {time.perf_counter() - tic:.2f}s")
    print(f"Number of paths retrieved: {len(all_node_path)}")
    print(f"Number of edges retrieved: {sum([len(p) - 1 for p in all_node_path])}")

    tic = time.perf_counter()
    all_paths = []
    all_nodes = set()
    for node_path in all_node_path:
        path_data = []
        for i in range(len(node_path) - 1):
            all_nodes.add(node_path[i])
            node_data = await kg_inst.get_node(node_path[i])
            edge_data = await kg_inst.get_edge(node_path[i], node_path[i + 1])
            assert node_data is not None and edge_data is not None
            path_data.extend([node_data["name"], edge_data["relation"]])
        all_nodes.add(node_path[-1])
        node_data = await kg_inst.get_node(node_path[-1])
        assert node_data is not None
        path_data.append(node_data["name"])
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
