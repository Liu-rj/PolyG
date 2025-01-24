import logging
import time
import boto3
from typing import List
import jsonlines
from collections import defaultdict
import tiktoken

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)


MAX_MODEL_LEN = 128000
MAX_CONTEXT_TOKENS = 100000
MAX_OUTPUT_TOKENS = 5000

CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

question_types = {
    "single_entity_abstract": 0,
    "single_entity_concrete": 1,
    "multi_entity_abstract": 2,
    "multi_entity_concrete": 3,
}


# prompt_ = """
# You are a highly intelligent assistant tasked with classifying questions into four types based on the number of entities mentioned and the nature of the aspect being asked.
# - single_entity_abstract (0): The question mentions one entity and asks about its own aspects.  It does not ask about other entities.
# - single_entity_concrete (1): The question mentions one entity and asks about its relations to other entities, or specific details  such as relationships, interactions, or comparisons.
# - multi_entity_abstract (2): The question mentions multiple entities and asks about their abstract relationships, concepts, or interactions.
# - multi_entity_concrete (3): The question mentions multiple entities and asks about concrete details or specific attributes of their relationship.
# When given a question, return only a single number corresponding to its type:
# - 0 for single_entity_abstract
# - 1 for single_entity_concrete
# - 2 for multi_entity_abstract
# - 3 for multi_entity_concrete
# Question: {}
# Your Answer:
# """

prompt = """
**Prompt:**
You are an intelligent assistant tasked with classifying questions into four types based on the number of entities mentioned and the nature of the aspect being asked.

- **single_entity_abstract (0):**
  The question focuses on one entity and asks about general or abstract information about it (e.g., themes, concepts, or a general description).
  Examples:
  - "Tell me about 'The Woman in Black: A Ghost Play'."
  - "Tell me about 'Frankenstein, or The Modern Prometheus'."
- **single_entity_concrete (1):**
  The question focuses on one entity and asks about specific, concrete details related to it (e.g., names, authors, publishers, or other direct attributes).
  Examples:
  - "Who are the authors of the book 'Sunshine for the Latter-Day Sa'?"
  - "What series have the author of the book 'Cookies for the Dragon (Saint Lakes, #2.1)' published?"
  - "What are the authors of the books that are published by the publishers that have published books of the series 'Shifter Justice'?"
- **multi_entity_abstract (2):**
  The question involves multiple entities and focuses on their abstract relationships or interactions (e.g., conceptual connections, influences).
  Examples:
  - "What is the relationship between 'Kizuna' and 'Kazuma Kodaka'?"
  - "What is the relationship between 'Monster House Press' and 'Matt Hart'?"
- **multi_entity_concrete (3):**
  The question involves multiple entities and focuses on concrete details related to their relationship (e.g., shared authors, publishers, or specific examples).
  Examples:
  - "Have the authors 'Rubem Fonseca' and 'Lygia Fagundes Telles' ever published books in the same publishers? If so, tell me some examples."
  - "Do the publishers 'Scholastic Inc.' and 'Klutz' have any authors publishing books in both of them and what are the publications and authors?"

**Instruction:**
When given a question, analyze it based on the definitions above. Then, return **only a single number** corresponding to the type of the question:

- **0** for single_entity_abstract
- **1** for single_entity_concrete
- **2** for multi_entity_abstract
- **3** for multi_entity_concrete

**Question:** {}

**Your Answer:**
"""


def num_tokens(text: str, token_encoder: tiktoken.Encoding | None = None) -> int:
    """Return the number of tokens in the given text."""
    if token_encoder is None:
        token_encoder = tiktoken.get_encoding("cl100k_base")
    return len(token_encoder.encode(text))


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


def load_data():
    questions = defaultdict(list)
    for qtype in question_types:
        for dataset in ["amazon", "goodreads", "physics"]:
            with open(f"../examples/benchmarks/{dataset}/{qtype}.jsonl", "r") as f:
                for line in jsonlines.Reader(f):
                    x = line["question"]
                    y = question_types[qtype]
                    questions[dataset].append((x, y))
    return questions


def main():
    questions = load_data()
    for dataset, q_list in questions.items():
        log = open(f"{dataset}_results.csv", "w")
        print(
            "num_tokens", "latency", "prediction", "type", "question", sep=",", file=log
        )
        for question, type in q_list:
            query = prompt.format(question)
            n_tok = num_tokens(query)
            start_time = time.time()
            response = bedrock_generator(query)
            end_time = time.time()
            print(
                n_tok,
                end_time - start_time,
                response.strip(),
                type,
                question,
                sep=",",
                file=log,
            )
            print(
                n_tok,
                end_time - start_time,
                response.strip(),
                type,
                question,
                sep="\t",
                flush=True,
            )
        log.close()


if __name__ == "__main__":
    main()
