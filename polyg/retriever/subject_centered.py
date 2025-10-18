import asyncio
import time
from typing import List, Tuple, Dict
from ..base import BaseGraphStorage, QueryParam, ID, RetrievalResult
from ..prompt import PROMPTS, SCHEMA_MAP
from ..utils import logger


async def bfs_retriever(
    query: str,
    id_mapping: dict[str, ID],
    kg_inst: BaseGraphStorage,
    query_param: QueryParam,
    global_config: dict,
) -> RetrievalResult:
    cypher_query = (
        "MATCH (s:self.namespace)-[r]-(t:self.namespace) WHERE s.id = $source_id\n"
        "RETURN s.id AS source, t.id AS target, Type(r) AS relation"
    )

    all_nodes = set()  # all sampled nodes
    seed_nodes = id_mapping.values()

    tic = time.perf_counter()
    logger.info(f"Number of nodes before traversal: {len(seed_nodes)}")
    logger.info(
        f"Traversal type: {query_param.traversal_type}, depth: {query_param.edge_depth}"
    )

    unique_edges = set()
    for _ in range(query_param.edge_depth):
        all_nodes.update(seed_nodes)
        related_edges = await asyncio.gather(
            *[kg_inst.get_node_edges(nid) for nid in seed_nodes]
        )

        next_frontiers = set()
        for edges in related_edges:
            sampled_nids = [e["src_id"] for e in edges] + [e["tgt_id"] for e in edges]
            next_frontiers.update([x for x in sampled_nids if x not in all_nodes])
            hashable_edges = [frozenset(e.items()) for e in edges]
            unique_edges.update(hashable_edges)

        seed_nodes = next_frontiers
        logger.info(f"Number of nodes: {len(all_nodes)}")
    all_nodes.update(seed_nodes)
    num_edges = len(unique_edges)

    logger.info(f"Number of nodes retrieved: {len(all_nodes)}")
    logger.info(f"Number of edges retrieved: {num_edges}")
    logger.info(f"Traversal time: {time.perf_counter() - tic:.2f}s")

    tic = time.perf_counter()
    nodes_data = await asyncio.gather(*[kg_inst.get_node(nid) for nid in all_nodes])
    assert all([n is not None for n in nodes_data])
    logger.info(f"Node fetching time: {time.perf_counter() - tic:.2f}s")

    return RetrievalResult(
        cypher_query=cypher_query,
        nodes_data=nodes_data,  # type: ignore
        edges_data=[dict(x) for x in unique_edges],
        reasoning_paths=[],
        auxiliary_data=[],
        used_tokens=0,
        num_llm_calls=0,
        answer_list=[],
    )
