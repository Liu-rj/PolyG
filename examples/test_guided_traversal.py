import logging
import boto3
from typing import List
from polyg.prompt import PROMPTS

logging.basicConfig(level=logging.WARNING)
logging.getLogger("polyg").setLevel(logging.INFO)


MAX_MODEL_LEN = 128000
MAX_CONTEXT_TOKENS = 100000
MAX_OUTPUT_TOKENS = 5000

CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"


single_entity_concrete_template = {
    "physics": {
        "author": {
            "What paper have the author '{}' published?": {
                "cypher": """
            MATCH (author:physics:author {{id: '{}'}})
            -[:paper]->(paper:physics:paper)
            RETURN paper.name as name
            """,
                "hops": 1,
            },
            "What are the academic collaborators of '{}'?": {
                "cypher": """
            MATCH (author:physics:author {{id: '{}'}})
            -[:paper]->(paper:physics:paper)
            -[:author]->(collaborator:physics:author)
            WHERE collaborator <> author
            RETURN DISTINCT collaborator.name AS name
            """,
                "hops": 2,
            },
            "What venues have the author '{}' published in?": {
                "cypher": """
            MATCH (author:physics:author {{id: '{}'}})
            -[:paper]->(paper:physics:paper)
            -[:venue]->(venue:physics:venue)
            RETURN DISTINCT venue.name AS name
            """,
                "hops": 2,
            },
            "Which authors have also published paper in the venues where the author '{}' have published in?": {
                "cypher": """
            MATCH (author:physics:author {{id: '{}'}})
            -[:paper]->(paper:physics:paper)
            -[:venue]->(venue:physics:venue)
            -[:paper]->(otherPaper:physics:paper)
            -[:author]->(otherAuthor:physics:author)
            WHERE otherAuthor <> author
            RETURN DISTINCT otherAuthor.name AS name
            """,
                "hops": 4,
            },
        },
        "paper": {
            "Who are the authors of the paper '{}'?": {
                "cypher": """
            MATCH (p:physics:paper {{id: '{}'}})
            -[:author]->(a:physics:author)
            RETURN DISTINCT a.name AS name
            """,
                "hops": 1,
            },
            "Who are the academic collaborators of the author who writes the paper '{}'?": {
                "cypher": """
            MATCH (p:physics:paper {{id: '{}'}})
            MATCH (p)-[:author]->(a:physics:author)
            MATCH (a)-[:paper]->(otherPaper:physics:paper)
            MATCH (otherPaper)-[:author]->(coAuthor:physics:author)
            WHERE coAuthor <> a
            RETURN DISTINCT coAuthor.name AS name
            """,
                "hops": 3,
            },
            "What venues have the author of the paper '{}' published in?": {
                "cypher": """
            MATCH (p:physics:paper {{id: '{}'}})
            -[:author]->(a:physics:author)
            -[:paper]->(other_p:physics:paper)
            -[:venue]->(v:physics:venue)
            RETURN DISTINCT v.name AS name
            """,
                "hops": 3,
            },
            "What venues have the academic collaborators of the author who writes the paper '{}' published in?": {
                "cypher": """
            MATCH (start_paper:physics:paper {{id: '{}'}})
            -[:author]->(author:physics:author)
            -[:paper]->(collab_paper:physics:paper)
            -[:author]->(collaborator:physics:author)
            WHERE collaborator <> author
            MATCH (collaborator)-[:paper]->(pub:physics:paper)
            -[:venue]->(venue:physics:venue)
            RETURN DISTINCT venue.name AS name
            """,
                "hops": 5,
            },
        },
    },
    "amazon": {
        "item": {
            # "What is the brand of the item '{}'?": {
            #     "cypher": """
            # MATCH (:amazon:item {{id: '{}'}})
            # -[:brand]->(brand:amazon:brand)
            # RETURN DISTINCT brand.name as name
            # """,
            #     "hops": 1,
            # },
            "What are the brands of the items that are also brought after viewing the item '{}'?": {
                "cypher": """
            """,
                "hops": 2,
            },
            # "What are the items that are also viewed when viewing items of the brand owning the item '{}'?": {
            #     "hops": 3,
            # },
            # "What are the brands of the items that are brought together with items of the brand owning the item '{}'?": {
            #     "hops": 4,
            # },
        },
        "brand": {
            # "What are the items of the brand '{}'?": {"hops": 1},
            # "What are the items that are also brought together with items of the brand '{}'?": {
            #     "hops": 2,
            # },
            # "What are the brands of the items that are also brought after viewing items of the brand '{}'?": {
            #     "hops": 3,
            # },
            # "What items does the brands of the items that are also viewed together with items of the brand '{}' have?": {
            #     "hops": 4
            # },
        },
    },
    "goodreads": {
        "book": {
            "Who are the authors of the book '{}'?": {
                "cypher": """
            MATCH (book:goodreads:book {{id: '{}'}})
            -[:author]->(author:goodreads:author)
            RETURN DISTINCT author.name as name
            """,
                "hops": 1,
            },
            "What series have the author of the book '{}' published?": {
                "cypher": """
            MATCH (book:goodreads:book {{id: '{}'}})
            MATCH (book)-[:author]->(author:goodreads:author)
            MATCH (author)-[:book]->(other_books:goodreads:book)
            MATCH (other_books)-[:series]->(series:goodreads:series)
            RETURN DISTINCT series.name as name
            """,
                "hops": 3,
            },
        },
        "author": {
            "What books has the author '{}' published?": {
                "cypher": """
            MATCH (author:goodreads:author {{id: '{}'}})
            -[:book]->(book:goodreads:book)
            RETURN DISTINCT book.name as name
            """,
                "hops": 1,
            },
            "What books have the collaborators of the author '{}' published?": {
                "cypher": """
            MATCH (author1:goodreads:author {{id: '{}'}})-[:book]->(book1:goodreads:book)
            MATCH (book1)-[:author]->(coauthor:goodreads:author)
            WHERE coauthor <> author1
            MATCH (coauthor)-[:book]->(other_book:goodreads:book)
            RETURN DISTINCT other_book.name as name
            """,
                "hops": 3,
            },
            "What are the series published by the publishers that have published books of the author '{}'?": {
                "cypher": """
            MATCH (author:goodreads:author {{id: '{}'}})
            -[:book]->(authorBook:goodreads:book)
            -[:publisher]->(publisher:goodreads:publisher)
            -[:book]->(publisherBook:goodreads:book)
            -[:series]->(series:goodreads:series)
            RETURN DISTINCT series.name as name
            """,
                "hops": 4,
            },
        },
        "publisher": {
            "What are the authors of the books published by the publisher '{}'?": {
                "cypher": """
            MATCH (:goodreads:publisher {{id: '{}'}})
            -[:book]->(b:goodreads:book)-[:author]->(a:goodreads:author)
            RETURN DISTINCT a.name as name
            """,
                "hops": 2,
            },
        },
        "series": {
            "Where does the books of the series '{}' published in?": {
                "cypher": """
            MATCH (s:goodreads:series {{id: '{}'}})
            MATCH (s)-[:book]->(b:goodreads:book)-[:publisher]->(p:goodreads:publisher)
            RETURN DISTINCT p.name as name
            """,
                "hops": 2,
            },
            "What are the authors of the books that are published by the publishers that have published books of the series '{}'?": {
                "cypher": """
            MATCH (s:goodreads:series {{id: '{}'}})
            -[:book]->(book1:goodreads:book)
            -[:publisher]->(p:goodreads:publisher)
            -[:book]->(book2:goodreads:book)
            -[:author]->(a:goodreads:author)
            RETURN DISTINCT a.name as name
            """,
                "hops": 4,
            },
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
    for dataset, entities in single_entity_concrete_template.items():
        if dataset != "amazon":
            continue
        if dataset == "physics":
            prompt = PROMPTS["cypher_query_prompt_physics"]
        elif dataset == "amazon":
            prompt = PROMPTS["cypher_query_prompt_amazon"]
        elif dataset == "goodreads":
            prompt = PROMPTS["cypher_query_prompt_goodreads"]
        else:
            raise ValueError(f"Dataset {dataset} not supported")
        for entity, questions in entities.items():
            for question, cypher_query in questions.items():
                print(f"Question: {question}")
                response = bedrock_generator(question, prompt)
                print_outputs(response)


if __name__ == "__main__":
    gen_paths()
