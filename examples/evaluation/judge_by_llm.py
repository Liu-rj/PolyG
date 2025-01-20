import boto3
import json
import jsonlines
import re
import argparse
from collections import defaultdict


argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="physics", required=True)
args = argparser.parse_args()


if args.dataset == "physics":
    ANSWER_PATH = [
        "/home/ubuntu/graphrag_planner/examples/results/Physics/results.jsonl",
        "/home/ubuntu/fast-graphrag/examples/results/Physics/results.jsonl",
        "/home/ubuntu/Graph-CoT/Graph-CoT/results/claude-3-5-sonnet/maple-Physics/results.jsonl",
    ]
    OUTPUT_FILE = (
        "/home/ubuntu/graphrag_planner/examples/results/Physics/judgements.jsonl"
    )
elif args.dataset == "amazon":
    ANSWER_PATH = [
        "/home/ubuntu/graphrag_planner/examples/results/amazon/results.jsonl",
        "/home/ubuntu/fast-graphrag/examples/results/amazon/results.jsonl",
        "/home/ubuntu/Graph-CoT/Graph-CoT/results/claude-3-5-sonnet/amazon/results.jsonl",
    ]
    OUTPUT_FILE = (
        "/home/ubuntu/graphrag_planner/examples/results/amazon/judgements.jsonl"
    )
elif args.dataset == "goodreads":
    ANSWER_PATH = [
        "/home/ubuntu/graphrag_planner/examples/results/goodreads/results.jsonl",
        "/home/ubuntu/fast-graphrag/examples/results/goodreads/results.jsonl",
        "/home/ubuntu/Graph-CoT/Graph-CoT/results/claude-3-5-sonnet/goodreads/results.jsonl",
    ]
    OUTPUT_FILE = (
        "/home/ubuntu/graphrag_planner/examples/results/goodreads/judgements.jsonl"
    )


CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

SYSTEM_ROLE = """
---Role---
You are an expert tasked with evaluating responses to the some questions based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
"""

PROMPT_WITH_GT = """
You will evaluate multiple responses to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the best response(s) and explain the reason for this decision. Then, select an overall winner based on these three criteria.

Note:
1. You will be provided with the ground-truth answers to each question, and you should evaluate the reponses based on the groun-truth answers. Good reponses should be consistent with the ground-truth answers.
2. There can be multiple winners for a question in the case where they all well answer the question regarding the criteria. You also need to give the reasons for this case.
3. Reponse like "there is no direct information for me to answer" or other forms that indicat it can not give answers to the question is not a valid answer as all questions are designed to ensure there is an answer. These kinds of reponses should be considered as a bad reponse.
4. Responses that are off-topic and irrelevant to the question should be considered as a bad response.

Question:
{query}

Ground Truth Answers:
{gt_answer}

Responses:
{answer}

Evaluate the above responses using the three criteria and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{{
    "question_type": "{question_type}",
    "question": "{query}",
    "Comprehensiveness": {{
        "Winner": "[Method name 1], [Method name 2], ...",
        "Explanation": "[Provide explanation here]"
    }},
    "Diversity": {{
        "Winner": "[Method name 1], [Method name 2], ...",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Winner": "[Method name 1], [Method name 2], ...",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Method name 1], [Method name 2], ...",
        "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
    }}
}}
"""

PROMPT_WITHOUT_GT = """
You will evaluate multiple responses to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the best answer(s) and explain the reason for this decision. Then, select an overall winner based on these three criteria.

Note:
1. There can be multiple winners for a question in the case where they all well answer the question regarding the criteria. You also need to give the reasons for this case.
2. Reponse like "there is no direct information for me to answer" or other forms that indicat it can not give answers to the question is not a valid answer as all questions are designed to ensure there is an answer. These kinds of reponses should be considered as a bad reponse.
3. Responses that are off-topic and irrelevant to the question should be considered as a bad response.

Question:
{query}

Responses:
{answer}

Evaluate the above responses using the three criteria and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{{
    "question_type": "{question_type}",
    "question": "{query}",
    "Comprehensiveness": {{
        "Winner": "[Method name 1], [Method name 2], ...",
        "Explanation": "[Provide explanation here]"
    }},
    "Diversity": {{
        "Winner": "[Method name 1], [Method name 2], ...",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Winner": "[Method name 1], [Method name 2], ...",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Method name 1], [Method name 2], ...",
        "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
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
    # "single_entity_abstract",
    # "single_entity_concrete",
    # "multi_entity_abstract",
    "multi_entity_concrete",
]
question_answer = {key: defaultdict(list) for key in question_types}
for item in answers:
    if item["question_type"] not in question_answer.keys():
        continue
    question_answer[item["question_type"]][item["question"]].append(
        (item["method"], item["model_answer"], item["gt_answer"])
    )

criteria = ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]
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
            method, answer, gt = answer_tuple
            answer_str += f"Answer {it + 1} (Method {method}): {answer}\n"

        sys_prompt = PROMPT_WITH_GT if gt != "N/A" else PROMPT_WITHOUT_GT

        result = bedrock_generator(
            prompt=sys_prompt.format(
                query=question,
                answer=answer_str,
                question_type=question_type,
                gt_answer=gt,
            ),
            system_prompt=SYSTEM_ROLE,
        )
        print(result)

        # Regular expression to extract the Explanation parts
        pattern1 = re.compile(r'"Explanation":\s*"(.*?)"\n')

        # Function to replace double quotes with single quotes in the explanation content
        def replace_double_quotes(match):
            content = match.group(1)
            modified_content = content.replace('"', "'").replace("\\'", "'")
            return f'"Explanation": "{modified_content}"\n'

        # Replace double quotes in the Explanation contents
        modified_json_str = pattern1.sub(replace_double_quotes, result)

        # convert str to dict
        result = json.loads(modified_json_str)

        for criterion in criteria:
            winner = result[criterion]["Winner"]
            for key in method_wins[criterion].keys():
                if key in winner:
                    method_wins[criterion][key] += 1

        with jsonlines.open(OUTPUT_FILE, "a") as writer:
            writer.write(result)

    all_method_wins[question_type] = method_wins
    print(method_wins)

for question_type, method_wins in all_method_wins.items():
    print(f"Question Type: {question_type}")
    print(method_wins)
