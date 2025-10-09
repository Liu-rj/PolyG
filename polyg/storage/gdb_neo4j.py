import json
import asyncio
import neo4j
from neo4j import AsyncGraphDatabase
from dataclasses import dataclass
from typing import Union, List, Any, Tuple, Set, Dict
from ..base import BaseGraphStorage, ID
from ..utils import logger
from contextlib import asynccontextmanager

neo4j_lock = asyncio.Lock()


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
            max_connection_pool_size=100,
            connection_timeout=3600,
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
                f"MATCH (s:{self.namespace})-[r]-(t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                "RETURN TYPE(r) AS edge_data",  # type: ignore
                source_id=src_id,
                target_id=tgt_id,
            )
            record = await result.single()
            return {"relation": record["edge_data"]} if record else None

    async def get_node_edges(self, node_id: ID) -> List[tuple[ID, ID, str]]:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]-(t:{self.namespace}) WHERE s.id = $source_id "
                "RETURN s.id AS source, t.id AS target, Type(r) AS relation",  # type: ignore
                source_id=node_id,
            )
            edges = []
            async for record in result:
                edges.append((record["source"], record["target"], record["relation"]))
            return edges

    async def index_done_callback(self):
        await self.async_driver.close()

    async def exec_query(self, query: str) -> List[Any]:
        result_list = []

        async with self.async_driver.session() as session:
            try:
                async with transaction_context(session, timeout=60) as tx:
                    results = await tx.run(query)

                    async for record in results:
                        result_list.append(record)
            except Exception as e:
                print(f"Error executing query: {e}")

            return result_list

    async def exec_query_and_get_path(
        self, query: str
    ) -> Tuple[List[str], Set[ID], Set[ID]]:
        paths, node_ids, dest_ids = [], set(), set()

        async with self.async_driver.session() as session:
            try:
                async with transaction_context(session, timeout=60) as tx:
                    results = await tx.run(query)

                    # Iterate through the results asynchronously
                    async for record in results:
                        for key in record.keys():
                            if "path" in key.lower():
                                path = record[key]
                                path_repr = []

                                # Process nodes and relationships in the path
                                for i, node in enumerate(path.nodes):
                                    node_ids.add(node["id"])  # Add node
                                    path_repr.append(node["name"])  # Add node name
                                    if i < len(path.relationships):
                                        rel = path.relationships[i]
                                        path_repr.append(f"({rel.type})")

                                # Join the path representation as a readable string
                                paths.append(" -> ".join(path_repr))

                            if "target" in key.lower():
                                dest = record[key]
                                dest_ids.add(dest["id"])  # Add target node
                                node_ids.add(dest["id"])  # Also add to node_ids
            except Exception as e:
                print(f"Error executing query: {e}")

            return paths, node_ids, dest_ids

    async def topk_shortest_paths(self, src_id: ID, tgt_id: ID) -> List[List[ID]]:
        paths = []

        async with self.async_driver.session() as session:
            try:
                async with transaction_context(session, timeout=60) as tx:
                    results = await tx.run(
                        f"""
                        MATCH p = SHORTEST 20 (s:{self.namespace} {{id: $source_id}})
                        -[*]->(t:{self.namespace} {{id: $target_id}})
                        RETURN [n in nodes(p) | n.id] AS path
                        """,
                        source_id=src_id,
                        target_id=tgt_id,
                    )

                    async for record in results:
                        node_id = record["path"]
                        paths.append(node_id)
            except Exception as e:
                print(f"Error executing query: {e}")

            return paths
