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
        # "Does authors '{}' cited or been cited by the work of '{}' and what are they?": {
        #     "cypher_template": """
        #     MATCH (author1:physics:author)
        #     -[:paper]->(paper1:physics:paper)
        #     -[:reference|cited_by]->(paper2:physics:paper)
        #     -[:author]->(author2:physics:author)
        #     WHERE author1 <> author2
        #     RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
        #     """,
        #     "cypher": """
        #     MATCH path = (author1:physics:author {{id: '{}'}})
        #     -[:paper]->(paper1:physics:paper)
        #     -[:reference|cited_by]->(paper2:physics:paper)
        #     -[:author]->(author2:physics:author {{id: '{}'}})
        #     RETURN path LIMIT 10
        #     """,
        #     "hops": 3,
        # },
        # "Have authors '{}' and '{}' both collaborated with some other authors? If so, tell me about them.": {
        #     "cypher_template": """
        #     MATCH (author1:physics:author)
        #     -[:paper]->(paper1:physics:paper)
        #     -[:author]->(collaborator:physics:author)
        #     -[:paper]->(paper2:physics:paper)
        #     -[:author]->(author2:physics:author)
        #     WHERE author1 <> collaborator AND author2 <> collaborator
        #     RETURN author1.name AS name1, author1.id AS id1, author2.name AS name2, author2.id AS id2
        #     """,
        #     "cypher": """
        #     MATCH path = (author1:physics:author {{id: '{}'}})
        #     -[:paper]->(paper1:physics:paper)
        #     -[:author]->(collaborator:physics:author)
        #     -[:paper]->(paper2:physics:paper)
        #     -[:author]->(author2:physics:author {{id: '{}'}})
        #     WHERE author1 <> collaborator AND author2 <> collaborator
        #     RETURN path LIMIT 10
        #     """,
        #     "hops": 4,
        # },
        "Does venue '{}' and venue '{}' have same authors publishing work in both of them and who are they?": {
            "cypher_template": """
            MATCH path = (author:physics:author)-[:paper]->(paper1:physics:paper)-[:venue]->(:physics:venue {id: 'venue_A_id'})
            WHERE (author)-[:paper]->(:physics:paper)-[:venue]->(:physics:venue {id: 'venue_B_id'})
            RETURN path
            LIMIT 10
            """,
            "cypher": "",
        },
        # "What is the collaboration relationship between the authors of the paper '{}' and '{}'?": {
        #     "cypher_template": """
        #     MATCH path = (paper1:physics:paper)-[:author]->(author1:physics:author)
        #     -[:paper]->(sharedPaper:physics:paper)<-[:paper]-(author2:physics:author)
        #     <-[:author]-(paper2:physics:paper)
        #     WHERE author1 <> author2
        #     RETURN paper1.name AS name1, paper1.id AS id1, paper2.name AS name2, paper2.id AS id2
        #     """,
        #     "cypher": "",
        # },
    },
    "amazon": {
        # "Have the items of the brands 'A' and 'B' ever been brought together, and if so, what are those items? A: 1, B, 2": {
        #     "hops": 3,
        # },
        # "Are there any brands whose items are also brought when buying the items 'A' and 'B'? If so, tell me about those brands and their items. A: 1, B, 2": {
        #     "hops": 4,
        # },
        # "Are there any brands whose items are also viewed when viewing the items 'A' and 'B' and what are those brands? A: 1, B, 2": {
        #     "hops": 4,
        # },
        # "Have the items of the brands 'A' and 'B' ever been viewed together with some other items, and if so, what are those items? A: 1, B, 2": {
        #     "hops": 4,
        # },
        # "Have the items of the brands '{}' and '{}' ever been bought together with some other items, and if so, what are those items? A: 1, B, 2": {}
        # "Are the items '{}' and '{}' both also bought with items of some other brands? If so, tell me about those brands and their items. A: 1, B, 2": {
        #     "cypher_template": """
        #     MATCH (itemA:amazon:item)-[:also_viewed_item]->(viewedItemA:amazon:item)-[:brand]->(brandA:amazon:brand)<-[:brand]-(viewedItemB:amazon:item)<-[:also_viewed_item]-(itemB:amazon:item)
        #     WHERE itemA <> itemB
        #     RETURN itemA.name AS name1, itemA.id AS id1, itemB.name AS name2, itemB.id AS id2
        #     """,
        #     "cypher": """
        #     MATCH path = (itemA:amazon:item {{id: '{}'}})-[:also_viewed_item]->(viewedItemA:amazon:item)-[:brand]->(brandA:amazon:brand)-[:item]->(viewedItemB:amazon:item)-[:also_viewed_item]->(itemB:amazon:item {{id: '{}'}})
        #     RETURN path LIMIT 10
        #     """,
        #     "hops": 4,
        # },
        # "Are the items '{}' and '{}' both also viewed with items of some other brands and what are those items? A: 1, B, 2": {
        #     "cypher_template": """
        #     MATCH (item1:amazon:item)-[:also_viewed_item]->(also_viewed1:amazon:item)-[:brand]->(brand:amazon:brand)<-[:brand]-(also_viewed2:amazon:item)<-[:also_viewed_item]-(item2:amazon:item)
        #     WHERE item1 <> item2
        #     RETURN item1.name AS name1, item1.id AS id1, item2.name AS name2, item2.id AS id2
        #     """,
        #     "cypher": """
        #     MATCH path = (item1:amazon:item {{id: '{}'}})-[:also_viewed_item]->(also_viewed1:amazon:item)-[:brand]->(brand:amazon:brand)-[:item]->(also_viewed2:amazon:item)-[:also_viewed_item]->(item2:amazon:item {{id: '{}'}})
        #     RETURN path LIMIT 10
        #     """,
        #     "hops": 4,
        # },
        "Have the items of the brands '{}' and '{}' ever both been also bought with some other items, and if so, what are those items? A: 1, B, 2": {
            "cypher_template": """
            MATCH (brandA:amazon:brand)-[:item]->(itemA:amazon:item)-[:also_viewed_item]->(viewedItem:amazon:item)<-[:also_viewed_item]-(itemB:amazon:item)<-[:item]-(brandB:amazon:brand)
            WHERE brandA <> brandB
            RETURN brandA.name AS name1, brandA.id AS id1, brandB.name AS name2, brandB.id AS id2
            """,
            "cypher": """
            MATCH path = (brandA:amazon:brand {{id: '{}'}})-[:item]->(itemA:amazon:item)-[:also_viewed_item]->(viewedItem:amazon:item)-[:also_viewed_item]->(itemB:amazon:item)-[:brand]->(brandB:amazon:brand {{id: '{}'}})
            RETURN path LIMIT 10
            """,
            "hops": 4,
        },
        "Have the items of the brands '{}' and '{}' ever both been bought after viewing some other items, and if so, what are those items? A: 1, B, 2": {
            "cypher_template": """
            MATCH (b1:amazon:brand)-[:item]->(itemA:amazon:item)-[:bought_together_item]->(otherItem:amazon:item)<-[:bought_together_item]-(itemB:amazon:item)<-[:item]-(b2:amazon:brand)
            WHERE b1 <> b2
            RETURN b1.name AS name1, b1.id AS id1, b2.name AS name2, b2.id AS id2
            """,
            "cypher": """
            MATCH path = (b1:amazon:brand {{id: '{}'}})-[:item]->(itemA:amazon:item)-[:bought_together_item]->(otherItem:amazon:item)-[:bought_together_item]->(itemB:amazon:item)-[:brand]->(b2:amazon:brand {{id: '{}'}})
            RETURN path LIMIT 10
            """,
            "hops": 4,
        },
    },
    "goodreads": {
        # "Does series '{}' and '{}' contains books that are published in the same publisher? If so, tell me abou them.": {
        #     "cypher_template": """
        #     """,
        #     "cypher": """
        #     """,
        #     "hops": 4,
        # },
        # "Are there authors who have published books in both the series '{}' and '{}' and what are they?": {
        #     "cypher_template": """
        #     """,
        #     "cypher": """
        #     """,
        #     "hops": 4,
        # },
        "Does publishers '{}' and '{}' have books that are belong to the same series, and if so, what are they?": {
            "cypher_template": """
            """,
            "cypher": """
            """,
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
        if dataset != "amazon":
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
            print(f"Question: {question}")
            response = bedrock_generator(question, prompt)
            print_outputs(response)


if __name__ == "__main__":
    gen_paths()
