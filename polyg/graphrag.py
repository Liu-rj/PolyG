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
    SCHEMA_MAP,
)
from .utils import num_tokens
from .op import (
    local_query,
    cypher_only,
    guided_walk,
    topk_csp,
)
from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
)
from .storage import Neo4jStorage
from .base import (
    BaseGraphStorage,
    QueryParam,
)
from .llm import LLM


@dataclass
class GraphRAG:
    dataset: str
    working_dir: str | None = None

    # LLM
    model: str = "openai/gpt-4o"
    model_max_token_size: int = 32768
    model_max_async: int = 16
    llm: LLM = field(init=False)
    model_func: Callable = field(init=False)

    # graph schema
    concrete_graph_schema: str = "null"

    # storage
    graph_storage_cls: Type[BaseGraphStorage] = Neo4jStorage

    # extension
    create_working_dir: bool = False

    # extension
    addon_params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.working_dir and self.create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm = LLM(self.model)
        self.model_func = limit_async_func_call(self.model_max_async)(self.llm.generate)

        self.entity_relation_graph = self.graph_storage_cls(
            namespace="", global_config=asdict(self)
        )

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

    def query(
        self, query: str, id_mapping: dict, param: QueryParam = QueryParam()
    ) -> tuple[str, float, int, int, List]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, id_mapping, param))

    async def aquery(
        self, query: str, id_mapping: dict, param: QueryParam = QueryParam()
    ) -> tuple[str, float, int, int, List]:
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

            if global_traversal_type is None:
                logger.error("Failed to determine the traversal type")
                return (
                    PROMPTS["fail_response"],
                    time.time() - start,
                    total_api_calls,
                    0,
                    [],
                )
        print(f"Question classification time: {time.time() - tic:.2f}s")
        print(f"Traversal type: {global_traversal_type}")

        query_plan_str, query_plan = None, None
        if global_traversal_type == "nested":
            query_plan_str, query_plan, steps, token_len = (
                await self.decompose_nested_query(query, param)
            )
            total_api_calls += 1
            total_tokens += token_len
            print(f"Query plan: \n{query_plan_str}")
            print(f"Num of all steps: {steps}")

            if query_plan_str is None:
                logger.error("Failed to decompose the query")
                return (
                    PROMPTS["fail_response"],
                    time.time() - start,
                    total_api_calls,
                    0,
                    [],
                )
        else:
            steps = 1

        history = []
        response, answer_list = "N/A", []
        for step in range(steps):
            if global_traversal_type == "nested":
                assert query_plan_str is not None and query_plan is not None
                subqueries, traversal_type, token_len = await self.instantiate_query(
                    global_question,
                    query_plan_str,
                    query_plan,
                    step,
                    history,
                    id_mapping,
                    param,
                )
                total_api_calls += 1
                total_tokens += token_len

                if subqueries is None:
                    logger.warning("Failed to instance a concrete query, skip")
                    continue
            else:
                traversal_type = global_traversal_type
                subqueries = {
                    "question1": {"question": global_question, "id_mapping": id_mapping}
                }

            param.traversal_type = traversal_type  # type: ignore
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
                elif traversal_type in ["BFS", "topk_shortest_paths"]:
                    func = local_query
                elif traversal_type == "cypher_only":
                    func = cypher_only
                else:
                    logger.error(f"Unsupported traversal type: {traversal_type}")
                    return (
                        PROMPTS["fail_response"],
                        time.time() - start,
                        total_api_calls,
                        0,
                        [],
                    )

                response, token_len, api_calls, answer_list = await func(
                    subquery,
                    sub_id_mapping,
                    self.entity_relation_graph,
                    param,
                    asdict(self),
                )

                total_api_calls += api_calls
                total_tokens += token_len
                id_mapping.update(sub_id_mapping)
                history.append({"question": subquery, "response": response})

        if global_traversal_type == "nested":
            # combine the steps and generate a final summary to the global question
            print("All history:", history)
            prompt = PROMPTS["nested_query_summarization"].format(
                question=global_question,
                query_plan=query_plan_str,
                history=history,
            )
            response = await self.model_func(prompt=prompt)
            token_len = num_tokens(prompt)
            total_api_calls += 1
            total_tokens += token_len

        duration = time.time() - start
        print(f"Query time: {duration:.2f}s")
        return response, duration, total_tokens, total_api_calls, answer_list

    async def determine_traversal_type(
        self, query: str, query_param: QueryParam
    ) -> Tuple[str | None, int]:
        """
        Determine the traversal type based on the query by LLM.
        """
        traversal_types = [
            "BFS",
            "cypher_query",
            "topk_shortest_paths",
            "cypher_path_search",
        ]
        use_model_func = self.model_func
        prompt = PROMPTS["question_classification"].format(query)

        response = await use_model_func(prompt=prompt)
        query_param.question_classification_result = response
        token_len = num_tokens(prompt)
        print(response)

        try:
            num = int(response.split(":")[0].strip("\n").strip())
            if num != -1:
                return traversal_types[num], token_len
            else:
                return "nested", token_len
        except Exception as e:
            logger.error(f"Error in determining traversal type: {e}")
            return None, token_len

    async def decompose_nested_query(
        self, query: str, query_param: QueryParam
    ) -> Tuple[str | None, List[Tuple[str, str]], int, int]:
        """
        Decompose the nested query into sub-queries.
        """
        graph_schema = SCHEMA_MAP[self.dataset]
        if self.dataset in ["webqsp", "cwq"]:
            graph_schema = graph_schema.format(
                schema=self.concrete_graph_schema, benchmark=self.dataset, example=""
            )

        total_tokens = 0
        history_msgs = []
        prompt = PROMPTS["nested_query_decomposition"].format(
            graph_schema=graph_schema, query=query
        )
        plan_str, plan = None, []
        response = None
        for i in range(query_param.failure_retries + 1):
            try:
                response = await self.model_func(
                    prompt=prompt, history_messages=history_msgs
                )
                plan_str = response.split("```")[1].strip("plan").replace("\n\n", "\n")
                plan = plan_str.strip("\n").split("\n")
                for i, step in enumerate(plan):
                    traversal = step.split(":")[0]
                    description = step[len(traversal) + 1 :]
                    plan[i] = (traversal.split(".")[1].strip(), description.strip())

                total_tokens += num_tokens(prompt)
                break
            except Exception as e:
                logger.error(f"Error in decomposing query: {e}")
                history_msgs.extend(
                    [
                        ("user", prompt),
                        ("assistant", response),
                    ]
                )
                prompt = PROMPTS["error_retry"].format(str(e))
        return plan_str, plan, len(plan), total_tokens

    async def merge_query(self, query_plan: List[Tuple[str, str]]):
        for i, step in enumerate(query_plan):
            if step[0] != "<s,p,*>":
                return None
        return "cypher_query"

    async def instantiate_query(
        self,
        global_query: str,
        query_plan_str: str,
        query_plan: List[Tuple[str, str]],
        step: int,
        history: List[str],
        id_mapping: dict[str, str],
        query_param: QueryParam,
    ) -> Tuple[Dict[str, Dict] | None, str, int]:
        """
        Instantiate the query based on the query plan and step.
        """
        mapping = {
            "<s,*,*>": "BFS",
            "<s,p,*>": "cypher_query",
            "<s,*,o>": "topk_shortest_paths",
            "<s,p,o>": "cypher_path_search",
        }
        # extract the traversal type from the query plan
        traversal_type = mapping[query_plan[step][0]]

        history_str = ""
        for i, h in enumerate(history):
            history_str += f"Response for step {i + 1}: {h}\n"

        total_tokens = 0
        history_msgs = []
        prompt = PROMPTS["nested_query_instantiation"].format(
            question=global_query,
            query_plan=query_plan_str,
            step=step + 1,
            history=history_str,
            mapping=id_mapping,
        )
        response, concrete_queries = None, None
        for i in range(query_param.failure_retries + 1):
            try:
                response = await self.model_func(
                    prompt=prompt, history_messages=history_msgs
                )

                q_str = response.split("```")[1].strip("json")
                concrete_queries = json.loads(q_str)
                for k, v in concrete_queries.items():
                    if isinstance(v["id_mapping"], str):
                        new_map = v["id_mapping"].replace("'", '"')
                        concrete_queries[k]["id_mapping"] = json.loads(new_map)

                total_tokens += num_tokens(prompt)
                break
            except Exception as e:
                logger.error(f"Error in instantiating query: {e}")
                history_msgs.extend(
                    [
                        ("user", prompt),
                        ("assistant", response),
                    ]
                )
                prompt = PROMPTS["error_retry"].format(str(e))

        return concrete_queries, traversal_type, total_tokens
