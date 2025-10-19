import asyncio
import time
import networkx as nx
from ..base import BaseGraphStorage, QueryParam, ID, RetrievalResult


async def bfs_ppr_retriever(
    query: str,
    id_mapping: dict[str, ID],
    kg_inst: BaseGraphStorage,
    query_param: QueryParam,
    global_config: dict,
) -> RetrievalResult:
    # start with BFS
    all_nodes = set()  # all sampled nodes
    seed_nodes = id_mapping.values()

    tic = time.perf_counter()
    print(f"Number of nodes before traversal: {len(seed_nodes)}")
    print(
        f"Traversal type: {query_param.traversal_type}, depth: {query_param.edge_depth}"
    )

    unique_edges = set()
    for _ in range(query_param.edge_depth):
        print(f"Number of seed nodes: {len(seed_nodes)}")
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
    all_nodes.update(seed_nodes)
    num_edges = len(unique_edges)

    print(f"Number of nodes retrieved: {len(all_nodes)}")
    print(f"Number of edges retrieved: {num_edges}")
    print(f"Traversal time: {time.perf_counter() - tic:.2f}s")

    # create a networkx directed graph
    tic = time.perf_counter()
    nx_g = nx.DiGraph()
    for nid in all_nodes:
        nx_g.add_node(nid)

    for edata in unique_edges:
        edata = dict(edata)
        nx_g.add_edge(edata["src_id"], edata["tgt_id"], **edata)

    print("# nodes:", nx_g.number_of_nodes())
    print("# edges:", nx_g.number_of_edges())
    print(f"graph is directed: {nx_g.is_directed()}")
    print(f"Build nx graph time: {time.perf_counter() - tic:.2f}s")

    # start PPR
    tic = time.perf_counter()
    personalized_vector = {nid: 1 / len(id_mapping) for nid in id_mapping.values()}
    pr = nx.pagerank(nx_g, alpha=0.85, personalization=personalized_vector)
    print(f"PPR time: {time.perf_counter() - tic:.2f}s")

    # select top entities and extract relations in between
    tic = time.perf_counter()
    top_entities = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:500]
    top_entities = [x[0] for x in top_entities]
    nx_subg = nx_g.subgraph(top_entities)
    sub_edges = nx_subg.edges(data=True)
    print("# nodes in subgraph:", nx_subg.number_of_nodes())
    print("# edges in subgraph:", nx_subg.number_of_edges())
    print(f"Induce subgraph time: {time.perf_counter() - tic:.2f}s")

    # get entity and relation data
    tic = time.perf_counter()
    nodes_data = await asyncio.gather(*[kg_inst.get_node(x) for x in top_entities])

    tic = time.perf_counter()
    edges_data = [edata for _, _, edata in sub_edges]
    print(f"Get edges data time: {time.perf_counter() - tic:.2f}s")

    return RetrievalResult(
        cypher_query="BFS + PPR based retrieval",
        nodes_data=nodes_data,  # type: ignore
        edges_data=edges_data,
        reasoning_paths=[],
        auxiliary_data=[],
        used_tokens=0,
        num_llm_calls=0,
        answer_list=[],
    )
