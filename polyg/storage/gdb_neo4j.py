import json
import asyncio
import neo4j
from neo4j import AsyncGraphDatabase
from dataclasses import dataclass
from typing import Union, List, Any, Tuple, Set, Dict
from ..base import BaseGraphStorage, ID
from ..utils import logger
from contextlib import asynccontextmanager


@asynccontextmanager
async def transaction_context(session, **kwargs):
    tx = await session.begin_transaction(**kwargs)
    try:
        yield tx
        await tx.commit()
    except Exception:
        await tx.rollback()
        raise


@dataclass
class Neo4jStorage(BaseGraphStorage):
    def __post_init__(self):
        self.neo4j_url = self.global_config["addon_params"].get("neo4j_url", None)
        self.neo4j_auth = self.global_config["addon_params"].get("neo4j_auth", None)
        self.namespace = self.global_config["dataset"]
        logger.info(f"Using the label {self.namespace} for Neo4j as identifier")
        if self.neo4j_url is None or self.neo4j_auth is None:
            raise ValueError("Missing neo4j_url or neo4j_auth in addon_params")
        self.async_driver = AsyncGraphDatabase.driver(
            self.neo4j_url,
            auth=self.neo4j_auth,
            max_connection_pool_size=3000,  # Increased pool size
            connection_timeout=300,  # 5 minutes for establishing new connections
            connection_acquisition_timeout=300,  # 5 minutes to acquire from pool
        )

    async def has_node(self, node_id: ID) -> bool:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id RETURN COUNT(n) > 0 AS exists",  # type: ignore
                node_id=node_id,
            )
            record = await result.single()
            return record["exists"] if record else False

    async def has_edge(self, src_id: ID, tgt_id: ID) -> bool:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]-(t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                "RETURN COUNT(r) > 0 AS exists",  # type: ignore
                source_id=src_id,
                target_id=tgt_id,
            )
            record = await result.single()
            return record["exists"] if record else False

    async def node_degree(self, node_id: ID) -> int:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id "
                f"RETURN COUNT {{(n)-[]-(:{self.namespace})}} AS degree",  # type: ignore
                node_id=node_id,
            )
            record = await result.single()
            return record["degree"] if record else 0

    async def edge_degree(self, src_id: ID, tgt_id: ID) -> int:
        async with self.async_driver.session() as session:
            result_src = await session.run(
                f"""
                MATCH (s:{self.namespace})-[r]-()
                WHERE s.id = $src_id
                RETURN COUNT(r) AS degree
                """,  # type: ignore
                src_id=src_id,
            )
            result_tgt = await session.run(
                f"""
                MATCH (t:{self.namespace})-[r]-()
                WHERE t.id = $tgt_id
                RETURN COUNT(r) AS degree
                """,  # type: ignore
                tgt_id=tgt_id,
            )
            record_src = await result_src.single()
            record_tgt = await result_tgt.single()
            degree_src = record_src["degree"] if record_src else 0
            degree_tgt = record_tgt["degree"] if record_tgt else 0
            return degree_src + degree_tgt

    async def get_node(self, node_id: ID) -> Union[Dict, None]:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id RETURN properties(n) AS node_data",  # type: ignore
                node_id=node_id,
            )
            record = await result.single()
            return record["node_data"] if record else None

    async def get_edge(self, src_id: ID, tgt_id: ID) -> Union[Dict, None]:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                "RETURN TYPE(r) AS edge_type, properties(r) AS edge_data",  # type: ignore
                source_id=src_id,
                target_id=tgt_id,
            )
            record = await result.single()
            if record is None:
                return None
            else:
                edge_data = record["edge_data"]
                edge_data["relation"] = record["edge_type"]
                edge_data["src_id"] = src_id
                edge_data["tgt_id"] = tgt_id
                return edge_data

    async def get_node_edges(self, node_id: ID) -> List[Dict]:
        in_edges = await self.get_node_in_edges(node_id)
        out_edges = await self.get_node_out_edges(node_id)
        return in_edges + out_edges

    async def get_node_in_edges(self, node_id: ID) -> List[Dict]:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) WHERE t.id = $target_id "
                "RETURN s.id AS sid, t.id AS tid, Type(r) AS edge_type, properties(r) AS edge_data",  # type: ignore
                target_id=node_id,
            )
            edges = []
            async for record in result:
                edge_data = record["edge_data"]
                edge_data["relation"] = record["edge_type"]
                edges.append(
                    {"src_id": record["sid"], "tgt_id": record["tid"], **edge_data}
                )
            return edges

    async def get_node_out_edges(self, node_id: ID) -> List[Dict]:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) WHERE s.id = $source_id "
                "RETURN s.id AS sid, t.id AS tid, Type(r) AS edge_type, properties(r) AS edge_data",  # type: ignore
                source_id=node_id,
            )
            edges = []
            async for record in result:
                edge_data = record["edge_data"]
                edge_data["relation"] = record["edge_type"]
                edges.append(
                    {"src_id": record["sid"], "tgt_id": record["tid"], **edge_data}
                )
            return edges

    async def exec_query(self, query: str) -> List[Any]:
        result_list = []

        async with self.async_driver.session() as session:
            async with transaction_context(session, timeout=60) as tx:
                results = await tx.run(query)

                async for record in results:
                    result_list.append(record)

            return result_list

    async def exec_query_and_get_path(
        self, query: str
    ) -> Tuple[List[List], Set[ID], Set[ID]]:
        all_paths, node_ids, dest_ids = [], set(), set()

        async with self.async_driver.session() as session:
            async with transaction_context(session, timeout=60) as tx:
                results = await tx.run(query)

                # Iterate through the results asynchronously
                async for record in results:
                    for key in record.keys():
                        if "path" in key.lower():
                            path = record[key]
                            path_list = []

                            # Process nodes and relationships in the path
                            for i, node in enumerate(path.nodes):
                                node_ids.add(node["id"])  # Add node
                                path_list.append(node["name"])  # Add node name
                                if i < len(path.relationships):
                                    rel = path.relationships[i]
                                    path_list.append(f"({rel.type})")

                            # Join the path representation as a readable string
                            all_paths.append(path_list)

                        if "target" in key.lower():
                            dest = record[key]
                            dest_ids.add(dest["id"])  # Add target node
                            node_ids.add(dest["id"])  # Also add to node_ids

            return all_paths, node_ids, dest_ids

    async def topk_shortest_paths(
        self, src_id: ID, tgt_id: ID, k: int = 20
    ) -> List[List[Tuple[ID, str, ID]]]:
        """
        Returns top-k shortest paths as a list of edge tuples.
        Each path is a list of (src_id, relation_type, tgt_id) tuples.

        Strategy: First try forward directed paths, then backward directed,
        finally undirected. This avoids duplicate semantic paths.
        """
        paths = []
        seen_node_sequences: Set[Tuple[ID, ...]] = set()

        def extract_path(path) -> List[Tuple[ID, str, ID]] | None:
            """Extract edge tuples from a path, deduplicating by node sequence."""
            # Get node sequence for deduplication
            node_seq = tuple(node["id"] for node in path.nodes)
            if node_seq in seen_node_sequences:
                return None
            seen_node_sequences.add(node_seq)

            edge_tuples = []
            for rel in path.relationships:
                edge_tuples.append(
                    (rel.start_node["id"], rel.type, rel.end_node["id"])
                )
            return edge_tuples

        async with self.async_driver.session() as session:
            async with transaction_context(session, timeout=60) as tx:
                # 1. Try forward directed: -[*]->
                results = await tx.run(
                    f"""
                    MATCH path = SHORTEST {k} (s:{self.namespace} {{id: $source_id}})
                    -[*]->(t:{self.namespace} {{id: $target_id}})
                    RETURN path
                    """,
                    source_id=src_id,
                    target_id=tgt_id,
                )
                async for record in results:
                    edge_tuples = extract_path(record["path"])
                    if edge_tuples is not None:
                        paths.append(edge_tuples)

                if len(paths) >= k:
                    return paths[:k]

                # 2. Try backward directed: <-[*]-
                results = await tx.run(
                    f"""
                    MATCH path = SHORTEST {k} (s:{self.namespace} {{id: $source_id}})
                    <-[*]-(t:{self.namespace} {{id: $target_id}})
                    RETURN path
                    """,
                    source_id=src_id,
                    target_id=tgt_id,
                )
                async for record in results:
                    edge_tuples = extract_path(record["path"])
                    if edge_tuples is not None:
                        paths.append(edge_tuples)

                if len(paths) >= k:
                    return paths[:k]

                # 3. Try undirected: -[*]- (last resort)
                results = await tx.run(
                    f"""
                    MATCH path = SHORTEST {k} (s:{self.namespace} {{id: $source_id}})
                    -[*]-(t:{self.namespace} {{id: $target_id}})
                    RETURN path
                    """,
                    source_id=src_id,
                    target_id=tgt_id,
                )
                async for record in results:
                    edge_tuples = extract_path(record["path"])
                    if edge_tuples is not None:
                        paths.append(edge_tuples)

            return paths[:k]
