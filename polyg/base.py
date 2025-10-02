from dataclasses import dataclass, field
from typing import Dict, Union, Literal, Any, List, TypeVar, Tuple, Set


@dataclass
class QueryParam:
    mode: Literal["local", "global", "naive"] = "global"
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    top_k: int = 20
    # naive search
    naive_max_token_for_text_unit = 12000
    # local search
    local_max_token_for_text_unit: int = 4000  # 12000 * 0.33
    local_max_token_for_local_context: int = 4800  # 12000 * 0.4
    local_max_token_for_community_report: int = 3200  # 12000 * 0.27
    local_token_ratio_for_node: float = 0.5
    local_token_ratio_for_edge: float = 0.5
    local_community_single_one: bool = False
    edge_depth: int = 1
    local_context_length: int = 10000
    traversal_type: Literal[
        "BFS",
        "topk_shortest_paths",
        "cypher_query",
        "cypher_path_search",
        "cypher_only",
        "adaptive",
    ] = "adaptive"
    question_classification_result: str | None = None
    # global search
    global_min_community_rating: float = 0
    global_max_consider_community: float = 512
    global_max_token_for_community_report: int = 16384
    global_special_community_map_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    failure_retries: int = 1


ID = TypeVar("ID")


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

    async def get_node_edges(self, node_id: ID) -> List[tuple[ID, ID, str]]:
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
    ) -> Tuple[List[str], Set[ID], Set[ID]]:
        raise NotImplementedError
