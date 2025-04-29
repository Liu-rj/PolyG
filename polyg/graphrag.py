import asyncio
import os
import networkx as nx
import tiktoken
import time
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast, Tuple
from tqdm import tqdm
from .prompt import (
    PROMPTS,
    PHYSICS_GRAPH_SCHEMA,
    AMAZON_GRAPH_SCHEMA,
    GOODREADS_GRAPH_SCHEMA,
)
from ._utils import num_tokens


from ._llm import (
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete,
)
from ._op import (
    chunking_by_token_size,
    extract_entities,
    generate_community_report,
    get_chunks,
    local_query,
    direct_cypher,
    guided_walk,
    topk_csp,
    batch_local_query,
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)


@dataclass
class GraphRAG:
    working_dir: str = field(
        default_factory=lambda: f"./polyg_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = False

    # text chunking
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # graph clustering
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # community reports
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    # text embedding
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    query_better_than_threshold: float = 0.2

    # LLM
    using_azure_openai: bool = False
    model_func: callable = gpt_4o_complete
    model_max_token_size: int = 32768
    model_max_async: int = 16

    # entity extraction
    entity_extraction_func: callable = extract_entities

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

        if self.using_azure_openai:
            # If there's no OpenAI API key, use Azure OpenAI
            if self.model_func == gpt_4o_complete:
                self.model_func = azure_gpt_4o_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info(
                "Switched the default openai funcs to Azure OpenAI if you didn't set any of it"
            )

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"name", "node_id"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

        self.model_func = limit_async_func_call(self.model_max_async)(
            partial(self.model_func, hashing_kv=self.llm_response_cache)
        )

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(
        self, query: str, id_mapping: dict, param: QueryParam = QueryParam()
    ) -> tuple[str, float, int, int, list]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, id_mapping, param))

    def batch_query(self, query: List[str], param: QueryParam = QueryParam()):
        if param.mode != "local":
            raise NotImplementedError("Batch query only support local mode")
        response = batch_local_query(
            query,
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.community_reports,
            self.text_chunks,
            param,
            asdict(self),
        )
        return response

    async def aquery(
        self, query: str, id_mapping: dict, param: QueryParam = QueryParam()
    ) -> tuple[str, float, int, int, list]:
        assert param.mode == "local", "Only local mode is supported"

        total_api_calls = 0
        total_tokens = 0
        global_traversal_type = param.traversal_type
        global_question = query
        start = time.time()

        tic = time.time()
        if global_traversal_type == "adaptive":
            global_traversal_type, token_len = await self.determine_traversal_type(
                query, param
            )
            total_api_calls += 1
            total_tokens += token_len
        print(f"Question classification time: {time.time() - tic:.2f}s")
        print(f"Traversal type: {global_traversal_type}")

        if global_traversal_type == "nested":
            query_plan, steps, token_len = await self.decompose_nested_query(query)
            total_api_calls += 1
            total_tokens += token_len
            print(f"Query plan: \n{'\n'.join(query_plan)}")
            print(f"Num of all steps: {steps}")
        else:
            steps = 1

        history = []
        for step in range(steps):
            if global_traversal_type == "nested":
                subqueries, traversal_type, token_len = await self.instantiate_query(
                    global_question, query_plan, step, history, id_mapping
                )
                total_api_calls += 1
                total_tokens += token_len
            else:
                traversal_type = global_traversal_type
                subqueries = {
                    "question1": {"question": global_question, "id_mapping": id_mapping}
                }

            param.traversal_type = traversal_type
            print(f"Step {step + 1}/{steps}, traversal_type: {traversal_type}")
            print(f"Step {step + 1}/{steps}, history: {history}")
            print(f"Step {step + 1}/{steps}, subqueries: {subqueries}")

            for query_dict in subqueries.values():
                subquery = query_dict["question"]
                sub_id_mapping = query_dict["id_mapping"]
                print(f"Subquery: {subquery}, id_mapping: {sub_id_mapping}")

                if traversal_type == "cypher_query":
                    func = guided_walk
                elif traversal_type == "cypher_path_search":
                    func = topk_csp
                elif traversal_type in ["BFS", "shortest_path", "all_shortest_paths"]:
                    func = local_query
                elif traversal_type == "direct_cypher":
                    func = direct_cypher
                else:
                    logger.error(f"Unsupported traversal type: {traversal_type}")
                    return (
                        PROMPTS["fail_response"],
                        time.time() - start,
                        total_api_calls,
                        0,
                        "N/A",
                    )

                response, token_len, api_calls, answer_list = await func(
                    subquery,
                    sub_id_mapping,
                    self.chunk_entity_relation_graph,
                    self.entities_vdb,
                    self.community_reports,
                    self.text_chunks,
                    param,
                    asdict(self),
                )

                total_api_calls += api_calls
                total_tokens += token_len
                id_mapping.update(sub_id_mapping)
                history.append({"question": subquery, "response": response})

        if global_traversal_type == "nested":
            # combine the steps and generate a final summary to the global question
            print("ALl history:", history)
            prompt = PROMPTS["nested_query_summarization"].format(
                question=global_question,
                query_plan=query_plan,
                history=history,
            )
            response = await self.model_func(prompt=prompt)
            token_len = num_tokens(prompt)
            total_api_calls += 1
            total_tokens += token_len

        duration = time.time() - start
        print(f"Query time: {duration:.2f}s")
        await self._query_done()
        return response, duration, total_tokens, total_api_calls, answer_list

    async def determine_traversal_type(
        self, query: str, query_param: QueryParam = None
    ) -> str:
        """
        Determine the traversal type based on the query by LLM.
        """
        traversal_types = [
            "BFS",
            "cypher_query",
            "all_shortest_paths",
            "cypher_path_search",
        ]
        use_model_func = self.model_func
        prompt = PROMPTS["question_classification"].format(query)

        response = await use_model_func(prompt=prompt)
        query_param.question_classification_result = response
        token_len = num_tokens(prompt)
        print(response)

        try:
            num = int(response.split(":")[0].split("\n")[0])
            if num != -1:
                return traversal_types[num], token_len
            else:
                return "nested", token_len
        except Exception as e:
            logger.error(f"Error in determining traversal type: {e}")
            return None, token_len

    async def decompose_nested_query(
        self, query: str
    ) -> List[Tuple[List[str], int, int]]:
        """
        Decompose the nested query into sub-queries.
        """
        if "Physics" in self.working_dir:
            graph_schema = PHYSICS_GRAPH_SCHEMA
        elif "amazon" in self.working_dir:
            graph_schema = AMAZON_GRAPH_SCHEMA
        elif "goodreads" in self.working_dir:
            graph_schema = GOODREADS_GRAPH_SCHEMA
        else:
            raise NotImplementedError
        prompt = PROMPTS["nested_query_decomposition"].format(
            graph_schema=graph_schema, query=query
        )
        response = await self.model_func(prompt=prompt)
        decompose_plan = response.split("```")[1].strip("plan").replace("\n\n", "\n")
        decompose_plan = decompose_plan.strip("\n").split("\n")
        return decompose_plan, len(decompose_plan), num_tokens(prompt)

    async def merge_query(self, query_plan: List[str]):
        for i, step in enumerate(query_plan):
            traversal_type = step.split(":")[0].split(".")[1].strip()
            if traversal_type != "<s,p,*>":
                return None
        return "cypher_query"

    async def instantiate_query(
        self,
        global_query: str,
        query_plan: List[str],
        step: int,
        history: List[str],
        id_mapping: dict[str, str],
    ) -> Tuple[Dict[str, Dict], str, int]:
        """
        Instantiate the query based on the query plan and step.
        """
        mapping = {
            "<s,*,*>": "BFS",
            "<s,p,*>": "cypher_query",
            "<s,*,o>": "all_shortest_paths",
            "<s,p,o>": "cypher_path_search",
        }
        # extract the traversal type from the query plan
        current_step = query_plan[step]
        traversal_type = mapping[current_step.split(":")[0].split(".")[1].strip()]

        history_str = ""
        for i, h in enumerate(history):
            history_str += f"Response for step {i + 1}: {h}\n"

        total_tokens = 0
        history_msgs = []
        prompt = PROMPTS["nested_query_instantiation"].format(
            question=global_query,
            query_plan=query_plan,
            step=step + 1,
            history=history_str,
            mapping=id_mapping,
        )
        for i in range(3):
            try:
                response = await self.model_func(
                    prompt=prompt, history_messages=history_msgs
                )

                concrete_queries = response.split("```")[1].strip("json")
                concrete_queries = json.loads(concrete_queries)
                for k, v in concrete_queries.items():
                    if isinstance(v["id_mapping"], str):
                        id_mapping = v["id_mapping"].replace("'", '"')
                        concrete_queries[k]["id_mapping"] = json.loads(id_mapping)

                total_tokens += num_tokens(prompt)
                break
            except Exception as e:
                logger.error(f"Error in instantiating query: {e}")
                history_msgs.extend(
                    [
                        {"role": "user", "content": [{"text": prompt}]},
                        {"role": "assistant", "content": [{"text": response}]},
                    ]
                )
                prompt = PROMPTS["error_retry"].format(str(e))
        return concrete_queries, traversal_type, total_tokens

    def insert_from_networkx_graph(self, graph: nx.Graph):
        loop = always_get_an_event_loop()
        loop.run_until_complete(self.ainsert_from_networkx_graph(graph))

    async def ainsert_from_networkx_graph(self, graph: nx.Graph):
        self.chunk_entity_relation_graph._graph = graph

        # ---------- build vdb
        nodes = await self.chunk_entity_relation_graph.nodes()
        all_entities_data = [
            self.chunk_entity_relation_graph.get_node_sync(nid) for nid in nodes
        ]
        data_for_vdb = {
            compute_mdhash_id(dp["name"], prefix="ent-"): {
                "content": (
                    dp["name"] + "; " + dp["abstract"]
                    if dp["node_type"] == "paper"
                    else dp["name"]
                ),
                "name": dp["name"],
                "node_id": id,
            }
            for id, dp in zip(nodes, all_entities_data)
        }
        await self.entities_vdb.upsert(data_for_vdb)

        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def insert_from_networkx_to_neo4j(self, graph: nx.Graph):
        loop = always_get_an_event_loop()
        loop.run_until_complete(self.ainsert_from_networkx_to_neo4j(graph))

    async def ainsert_from_networkx_to_neo4j(self, graph: nx.Graph):
        # ---------- build neo4j from networkx
        # get all nodes and edges from the networkx graph
        all_nodes = list(graph.nodes())
        all_entities_data = [graph.nodes.get(nid) for nid in all_nodes]
        all_edges = list(graph.edges())
        all_edges_data = [graph.edges.get((e[0], e[1])) for e in all_edges]

        batch_size = 100
        num_batches_nodes = len(all_nodes) // batch_size + 1
        for i in tqdm(
            range(0, len(all_nodes), batch_size), total=num_batches_nodes, ncols=100
        ):
            node_ids = all_nodes[i : i + batch_size]
            nodes_data = all_entities_data[i : i + batch_size]
            await asyncio.gather(
                *[
                    self.chunk_entity_relation_graph.upsert_node_without_check(nid, dp)
                    for nid, dp in zip(node_ids, nodes_data)
                ]
            )

        num_batch_edges = len(all_edges) // batch_size + 1
        for i in tqdm(
            range(0, len(all_edges), batch_size), total=num_batch_edges, ncols=100
        ):
            edge_ids = all_edges[i : i + batch_size]
            edges_data = all_edges_data[i : i + batch_size]
            await asyncio.gather(
                *[
                    self.chunk_entity_relation_graph.upsert_edge_without_check(
                        e[0], e[1], dp
                    )
                    for e, dp in zip(edge_ids, edges_data)
                ]
            )

        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def ainsert(self, string_or_strings):
        await self._insert_start()
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            # ---------- new docs
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # ---------- chunking

            inserting_chunks = get_chunks(
                new_docs=new_docs,
                chunk_func=self.chunk_func,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size,
            )

            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # TODO: no incremental update for communities now, so just drop all
            await self.community_reports.drop()

            # ---------- extract/summary entity and upsert to graph
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await self.entity_extraction_func(
                inserting_chunks,
                knwoledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg
            # ---------- update clusterings of graph
            logger.info("[Community Report]...")
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm
            )
            await generate_community_report(
                self.community_reports, self.chunk_entity_relation_graph, asdict(self)
            )

            # ---------- commit upsertings and indexing
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
