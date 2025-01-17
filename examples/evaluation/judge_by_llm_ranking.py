import boto3
import json
import jsonlines
import re
from collections import defaultdict


# ANSWER_PATH = [
#     "/home/ubuntu/graphrag_planner/examples/results/Physics/results.jsonl",
#     "/home/ubuntu/fast-graphrag/examples/results/Physics/results.jsonl",
#     "/home/ubuntu/Graph-CoT/Graph-CoT/results/claude-3-5-sonnet/maple-Physics/results.jsonl",
# ]
# OUTPUT_FILE = (
#     "/home/ubuntu/graphrag_planner/examples/results/Physics/judgements_ranking.jsonl"
# )

ANSWER_PATH = [
    "/home/ubuntu/graphrag_planner/examples/results/goodreads/results.jsonl",
    "/home/ubuntu/fast-graphrag/examples/results/goodreads/results.jsonl",
    "/home/ubuntu/Graph-CoT/Graph-CoT/results/claude-3-5-sonnet/goodreads/results.jsonl",
]
OUTPUT_FILE = (
    "/home/ubuntu/graphrag_planner/examples/results/goodreads/judgements_ranking.jsonl"
)

CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

SYSTEM_ROLE = """
---Role---
You are an expert tasked with evaluating answers to the some questions based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
"""

PROMPT = """
You will evaluate multiple answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, give a ranking of the reponses and explain the reason for this decision. Then, give an overall ranking based on these three criteria.

Note:
1. The lower rank (The earlier position in the list), the better. For example, rank 1 is the best and rank 2 is the second best.
2. Different reponses can have the same ranking for a question in the case where they all answer the question similarly regarding the criteria. You also need to give the reasons for this case. Return the ranking in a list format, and methods with the same ranking should be seperated with ";".
3. Reponse like "there is no direct information for me to answer" or other forms that indicat it can not give answers to the question is not a valid answer as all questions are designed to ensure there is an answer. These kinds of answers should be considered as a bad answer.

Question:
{query}

Answers:
{answer}

Evaluate the above answers using the three criteria and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{{
    "question_type": "{question_type}",
    "question": "{query}",
    "Comprehensiveness": {{
        "Ranking": "Method name 1;Method name 2, Method name 3, ...",
        "Explanation": "[Provide explanation here]"
    }},
    "Diversity": {{
        "Ranking": "Method name 1;Method name 2, Method name 3, ...",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Ranking": "Method name 1;Method name 2, Method name 3, ...",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall": {{
        "Ranking": "Method name 1;Method name 2, Method name 3, ...",
        "Explanation": "[Summarize why this is the overall ranking based on the three criteria]"
    }}
}}
"""


def bedrock_generator(
    prompt: str,
    system_prompt: str = None,
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


answers = []
for path in ANSWER_PATH:
    with open(path, "r") as f:
        for item in jsonlines.Reader(f):
            answers.append(item)

question_types = [
    "single_entity_abstract",
    "single_entity_concrete",
    "multi_entity_abstract",
    "multi_entity_concrete",
]
question_answer = {key: defaultdict(list) for key in question_types}
for item in answers:
    if item["question_type"] not in question_answer.keys():
        continue
    question_answer[item["question_type"]][item["question"]].append(
        (item["method"], item["model_answer"])
    )

criteria = ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]
method_names = [
    "BFS",
    "shortest_paths",
    "cypher_single_entity",
    "cypher_multi_entity",
    "Fastgraphrag_PPR",
    "GraphCoT",
]
all_method_wins = {}
for question_type in question_types:
    method_wins = {name: {method: 0 for method in method_names} for name in criteria}
    for it, (question, answers) in enumerate(question_answer[question_type].items()):
        question = question.replace('"', "'")
        print(f"Question {it + 1}: {question}")

        answer_str = "Answers:\n"
        for it, answer_tuple in enumerate(answers):
            method, answer = answer_tuple
            answer_str += f"Answer {it + 1} (Method {method}): {answer}\n"

        result = bedrock_generator(
            prompt=PROMPT.format(
                query=question, answer=answer_str, question_type=question_type
            ),
            system_prompt=SYSTEM_ROLE,
        )
        print(result)

        # Regular expression to extract the Explanation parts
        pattern1 = re.compile(r'"Explanation":\s*"(.*?)"\n')

        # Function to replace double quotes with single quotes in the explanation content
        def replace_double_quotes(match):
            content = match.group(1)
            modified_content = content.replace('"', "'")
            return f'"Explanation": "{modified_content}"\n'

        # Replace double quotes in the Explanation contents
        modified_json_str = pattern1.sub(replace_double_quotes, result)

        # convert str to dict
        result = json.loads(modified_json_str)

        for criterion in criteria:
            ranking = result[criterion]["Ranking"].split(",")
            for rank, methods in enumerate(ranking):
                for key in method_wins[criterion].keys():
                    if key in methods:
                        method_wins[criterion][key] += rank + 1

        with jsonlines.open(OUTPUT_FILE, "a") as writer:
            writer.write(result)

    all_method_wins[question_type] = method_wins
    print(method_wins)

for question_type, method_wins in all_method_wins.items():
    print(f"Question Type: {question_type}")
    print(method_wins)
