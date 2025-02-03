import json
import boto3
from typing import List

CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

PROMPT = """
**Prompt:**
You are an intelligent assistant tasked with classifying questions into different types based on the missing parts of a fact and the nature of the aspect being asked.

- **single_entity_abstract[<s,p_a,?>] (0):**
  In this type of question, the object is missing in the facts and question asks about the object, which is general or abstract information of the subject (e.g., themes, concepts, or a general description) with a abstract predicate p_a.
  Examples:
  - "Tell me about 'The Woman in Black: A Ghost Play'."
  - "Tell me about 'Frankenstein, or The Modern Prometheus'."
- **single_entity_concrete[<s,p_c,?>] (1):**
  In this type of question, the object is missing in the facts and question focuses on finding the object, which is some specific, concrete details related to the subject (e.g., names, authors, publishers, or other direct attributes) with a concrete predicate p_c.
  Examples:
  - "Who are the authors of the book 'Sunshine for the Latter-Day Sa'?"
  - "What series have the author of the book 'Cookies for the Dragon (Saint Lakes, #2.1)' published?"
  - "What are the authors of the books that are published by the publishers that have published books of the series 'Shifter Justice'?"
- **multi_entity_abstract[<s,?_a,o>] (2):**
  In this question, both the subject and object is given but the predicate is missing. The question asks about their relationships or interactions (e.g., conceptual connections, influences) with no constraints on the relation types.
  Examples:
  - "What is the relationship between 'Kizuna' and 'Kazuma Kodaka'?"
  - "What is the relationship between 'Monster House Press' and 'Matt Hart'?"
- **multi_entity_concrete[<s,?_c,o>] (3):**
  In this question, both the subject and object is given and the predicates is partially given with constraints. The question asks about concrete details related to their some relationships (e.g., shared authors, publishers, or specific examples) by giving concrete constraints on the relation type.
  Examples:
  - "Have the authors 'Rubem Fonseca' and 'Lygia Fagundes Telles' ever published books in the same publishers? If so, tell me some examples."
  - "Do the publishers 'Scholastic Inc.' and 'Klutz' have any authors publishing books in both of them and what are the publications and authors?"

**Instruction:**
When given a question, analyze it based on the definitions above. If the question belongs to any of the four types, then return **only a single number** corresponding to the type of the question:

- **0** for single_entity_abstract
- **1** for single_entity_concrete
- **2** for multi_entity_abstract
- **3** for multi_entity_concrete

Note that the question may not belong to any of the types, in which case you should return **-1**.
For thoses cases, if they can be decomposed into multiple step where some step the sub-question belongs to one of the four types above and other steps are non-traversal operators like "comparing entity attributes" or "doing a sorting", then you should also provide a explanation for this on how it can be decomposed and return as follows:
- **-1: [Explanation of decomposition]**

For example:
- "What are the 5 biggest cities in the usa?": 1. the first step is a <s,p_c,?> question, traverse from the USA and find all the cities. 2. non-traversal operator: get the population of the city and do a sorting.
- "During what war did abraham lincoln serve as president?": 1. the first step is a <s,?_c,o> question, find the links between Lincoln and the president. 2. non-traversal operator: get the time slot (maybe on the edge attributes). 3. non-traversal operator: scan all wars and filter those in the time slot.
- "which city held the summer olympics twice?": 1. the first step is a <s,p_c,?> question, traverse from summer olympics to the cities. 2. non-traversal operator: count which cities appear twice in the results.

**You don't need to give explanation when the question falls into the above four types, only give explanation when it does not.**

**Question:** {}

**Your Answer:**
"""


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


# Load the JSON file
file_path = "datasets/WebQSP.train.json"  # Replace with your file path
with open(file_path, "r") as file:
    data = json.load(file)

# Extract questions
questions = data["Questions"]
print(f"Number of questions: {len(questions)}")

output_file = "results/WebQSP_result.jsonl"

for question in questions:
    q = question["ProcessedQuestion"]
    query = PROMPT.format(q)
    response = bedrock_generator(query)
    print(f"Question: {q}")
    print(f"Response: {response}")
    print("")
    # Write to file
    with open(output_file, "a") as file:
        file.write(json.dumps({"response": response, "question": q}) + "\n")
