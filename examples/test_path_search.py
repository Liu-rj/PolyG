import logging
import boto3
from typing import List
from nano_graphrag.prompt import PROMPTS

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)


MAX_MODEL_LEN = 128000
MAX_CONTEXT_TOKENS = 100000
MAX_OUTPUT_TOKENS = 5000

CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"


multi_entity_concrete_template = {
    "physics": {
        "What is the relationship between authors '{}' and '{}' regarding collaborated papers?": {
            "cypher": """
            MATCH (author1:Physics:author)
            -[:paper]->(paper1:Physics:paper)
            -[:author]->(author2:Physics:author)
            WHERE author1 <> author2
            RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
            """,
            "hops": 2,
        },
        "What is the relationship between authors '{}' and '{}' regarding paper references?": {
            "cypher": """
            MATCH (author1:Physics:author)
            -[:paper]->(paper1:Physics:paper)
            -[:reference|cited_by]->(paper2:Physics:paper)
            -[:author]->(author2:Physics:author)
            WHERE author1 <> author2
            RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
            """,
            "hops": 3,
        },
        "What is the relationship between authors authors '{}' and '{}' regarding common collaborators?": {
            "cypher": """
            MATCH (author1:Physics:author)
            -[:paper]->(paper1:Physics:paper)
            -[:author]->(collaborator:Physics:author)
            -[:paper]->(paper2:Physics:paper)
            -[:author]->(author2:Physics:author)
            WHERE author1 <> collaborator AND author2 <> collaborator
            RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
            """,
            "hops": 4,
        },
        "What is the relationship between authors '{}' and '{}' regarding common venues they have published in?": {
            "cypher": """
            MATCH (author1:Physics:author)
            -[:paper]->(paper1:Physics:paper)
            -[:venue]->(venue:Physics:venue)
            -[:paper]->(paper2:Physics:paper)
            -[:author]->(author2:Physics:author)
            WHERE author1 <> author2
            RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
            """,
            "hops": 4,
        },
    },
    "amazon": {
        "What is the relationship between items '{}' and '{}' regarding common brands?": {
            "hops": 2
        },
        "What is the relationship between brands '{}' and '{}' regarding item purchasing?": {
            "hops": 3
        },
        "What is the relationship between items '{}' and '{}' regarding brands that are purchased together?": {
            "hops": 4
        },
        "What is the relationship between brands '{}' and '{}' regarding commonly viewed brands?": {
            "hops": 6
        },
    },
    "goodreads": {
        "What is the relationship between authors '{}' and '{}' regarding collaborated books?": {
            "hops": 2,
        },
        "What is the relationship between authors '{}' and '{}' regarding common series?": {
            "hops": 4,
        },
        "What is the relationship between authors '{}' and '{}' regarding common publishers?": {
            "hops": 4,
        },
        "What is the relationship between publishers '{}' and '{}' regarding commonly involved authors?, ID: A and B": {
            "hops": 4,
        },
    },
}


def print_outputs(outputs):
    print("=" * 80)
    print("Generated reponse:")
    print(outputs)
    print("-" * 80)


def bedrock_generator(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[dict] = [],
    **kwargs,
) -> str:
    bedrock_cli = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
    )

    messages, system = [], []
    if system_prompt:
        system.append({"text": system_prompt})

    messages.append({"role": "user", "content": [{"text": prompt}]})

    response = bedrock_cli.converse(
        modelId=CHAT_MODEL_ID, messages=messages, system=system
    )
    return response["output"]["message"]["content"][0]["text"]


def gen_paths():
    for dataset, entities in multi_entity_concrete_template.items():
        if dataset != "physics":
            continue
        if dataset == "physics":
            prompt = PROMPTS["cypher_path_search_prompt_physics"]
        elif dataset == "amazon":
            prompt = PROMPTS["cypher_path_search_prompt_amazon"]
        elif dataset == "goodreads":
            prompt = PROMPTS["cypher_path_search_prompt_goodreads"]
        else:
            raise ValueError(f"Dataset {dataset} not supported")
        for question, contents in entities.items():
            question = "What is the relationship between authors authors 'David A. Williams' and 'Celine Peroux' regarding common collaborators?, 'David A. Williams': '2996670930', 'Celine Peroux': '3179571918'"
            print(f"Question: {question}")
            response = bedrock_generator(question, prompt)
            print_outputs(response)
            exit()


if __name__ == "__main__":
    gen_paths()
