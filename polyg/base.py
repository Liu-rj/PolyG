from dataclasses import dataclass, field
from typing import Dict, Union, Literal, Any, List, TypeVar, Tuple, Set


@dataclass
class QueryParam:
    mode: Literal["local", "global", "naive"] = "global"
    response_type: str = "Multiple Paragraphs"
    # local search
    token_ratio_for_node: float = 0.5
    token_ratio_for_edge: float = 0.4
    token_ratio_for_reasoning_path: float = 1
    token_ratio_for_auxiliary_data: float = 1
    edge_depth: int = 1
    local_context_length: int = 10000
    traversal_type: str = "adaptive"
    question_classification_result: str | None = None
    # failure self-correction
    failure_retries: int = 1
    # additional field for future use
    extra_data: Dict = field(default_factory=dict)


ID = TypeVar("ID")


@dataclass
class RetrievalResult:
    cypher_query: str
    nodes_data: List[Dict]
    edges_data: List[Dict]
    reasoning_paths: List
    auxiliary_data: List[Dict]
    used_tokens: int
    num_llm_calls: int
    answer_list: List


@dataclass
class BaseGraphStorage:
    namespace: str
    global_config: dict

    async def has_node(self, node_id: ID) -> bool:
        raise NotImplementedError

    async def has_edge(self, src_id: ID, tgt_id: ID) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: ID) -> int:
        raise NotImplementedError

    async def edge_degree(self, src_id: ID, tgt_id: ID) -> int:
        raise NotImplementedError

    async def get_node(self, node_id: ID) -> Union[Dict, None]:
        raise NotImplementedError

    async def get_edge(self, src_id: ID, tgt_id: ID) -> Union[Dict, None]:
        raise NotImplementedError

    async def get_node_edges(self, node_id: ID) -> List[Dict]:
        raise NotImplementedError

    async def get_node_in_edges(self, node_id: ID) -> List[Dict]:
        raise NotImplementedError

    async def get_node_out_edges(self, node_id: ID) -> List[Dict]:
        raise NotImplementedError

    async def topk_shortest_paths(self, src_id: ID, tgt_id: ID) -> List[List[ID]]:
        raise NotImplementedError

    async def nodes(self) -> list[ID]:
        raise NotImplementedError

    async def edges(self) -> list[tuple[ID, ID]]:
        raise NotImplementedError

    async def exec_query(self, query: str) -> List[Any]:
        raise NotImplementedError

    async def exec_query_and_get_path(
        self, query: str
    ) -> Tuple[List[List], Set[ID], Set[ID]]:
        raise NotImplementedError
