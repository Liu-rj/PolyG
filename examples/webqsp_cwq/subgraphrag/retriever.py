import asyncio
import time
import torch
import networkx as nx
from model import Retriever
from polyg.base import BaseGraphStorage, QueryParam, ID, RetrievalResult


def prepare_sample(device, sample):
    (
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot,
        target_triple_probs,
        a_entity_id_list,
        text_entity_list,
        non_text_entity_list,
        relation_list,
    ) = sample

    h_id_tensor = h_id_tensor.to(device)
    r_id_tensor = r_id_tensor.to(device)
    t_id_tensor = t_id_tensor.to(device)
    q_emb = q_emb.to(device)
    entity_embs = entity_embs.to(device)
    relation_embs = relation_embs.to(device)
    topic_entity_one_hot = topic_entity_one_hot.to(device)

    return (
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot,
        target_triple_probs,
        a_entity_id_list,
        text_entity_list,
        non_text_entity_list,
        relation_list,
    )


async def subgraphrag_retriever(
    query: str,
    id_mapping: dict[str, ID],
    kg_inst: BaseGraphStorage,
    query_param: QueryParam,
    global_config: dict,
) -> RetrievalResult:
    extra_data = query_param.extra_data
    model: Retriever = extra_data["model"]
    sample = extra_data["sample"]
    device = extra_data["device"]
    topk = extra_data["topk"]

    (
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot,
        target_triple_probs,
        a_entity_id_list,
        text_entity_list,
        non_text_entity_list,
        relation_list,
    ) = prepare_sample(device, sample)

    pred_triple_logits = model(
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot,
    )
    pred_triple_scores = torch.sigmoid(pred_triple_logits).reshape(-1)
    top_K_results = torch.topk(pred_triple_scores, min(topk, len(pred_triple_scores)))
    top_K_scores = top_K_results.values.cpu().tolist()
    top_K_triple_IDs = top_K_results.indices.cpu().tolist()

    entity_list = text_entity_list + non_text_entity_list
    all_nodes = set()
    edges_data = []
    for j, triple_id in enumerate(top_K_triple_IDs):
        all_nodes.add(entity_list[h_id_tensor[triple_id].item()])
        all_nodes.add(entity_list[t_id_tensor[triple_id].item()])
        edges_data.append(
            {
                "src_id": entity_list[h_id_tensor[triple_id].item()],
                "tgt_id": entity_list[t_id_tensor[triple_id].item()],
                "relation": relation_list[r_id_tensor[triple_id].item()],
                "score": top_K_scores[j],
            }
        )

    nodes_data = [{"id": nid, "name": nid} for nid in all_nodes]

    return RetrievalResult(
        cypher_query="",
        nodes_data=nodes_data,  # type: ignore
        edges_data=edges_data,
        reasoning_paths=[],
        auxiliary_data=[],
        used_tokens=0,
        num_llm_calls=0,
        answer_list=[],
    )
