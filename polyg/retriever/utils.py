import asyncio
from typing import List
from ..base import ID, BaseGraphStorage
from ..utils import enclose_string_with_quotes, list_to_csv


async def build_schema_example(kg_inst: BaseGraphStorage, node_ids: List[ID]):
    all_edges = set()
    related_edges = await asyncio.gather(
        *[kg_inst.get_node_edges(node_id) for node_id in node_ids]
    )
    for this_edges in related_edges:
        v_edges = [
            (edge["src_id"], edge["tgt_id"], edge["relation"]) for edge in this_edges
        ]
        all_edges.update(v_edges)

    relation_header = ",\t".join(
        [
            f"{enclose_string_with_quotes(data)}"
            for data in ["id", "source", "target", "relation"]
        ]
    )
    relations_section_list = [relation_header]
    for i, (sid, tid, relation) in enumerate(all_edges):
        raw_data = [i, sid, tid, relation]
        relations_section_list.append(
            ",\t".join([f"{enclose_string_with_quotes(data)}" for data in raw_data])
        )
    return list_to_csv(relations_section_list[:100])
