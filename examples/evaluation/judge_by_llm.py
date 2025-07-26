import boto3
import json
import jsonlines
import re
import argparse
import os
from openai import OpenAI
from collections import defaultdict
from dotenv import load_dotenv
from typing import List, Tuple

load_dotenv("../.env")


argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="physics", required=True)
argparser.add_argument("--model", type=str, default="claude-3.5-sonnet", required=False)
args = argparser.parse_args()


# CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
# CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
MODEL_ID = "us.deepseek.r1-v1:0"
bedrock_cli = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
# client = OpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com",
# )
client = OpenAI()


if args.dataset == "physics":
    ANSWER_PATH = [
        f"/home/ubuntu/PolyG/examples/results/Physics/{args.model}/results_rephrased.jsonl",
        f"/home/ubuntu/fast-graphrag/examples/results/Physics/{args.model}/results_rephrased.jsonl",
        f"/home/ubuntu/Graph-CoT/Graph-CoT/results/{args.model}/maple-Physics/results_rephrased.jsonl",
    ]
    OUTPUT_FILE = f"/home/ubuntu/PolyG/examples/results/Physics/{args.model}/judgements_rephrased.jsonl"
elif args.dataset == "amazon":
    ANSWER_PATH = [
        f"/home/ubuntu/PolyG/examples/results/amazon/{args.model}/results_rephrased_sp*_nested.jsonl",
        f"/home/ubuntu/fast-graphrag/examples/results/amazon/{args.model}/results_rephrased.jsonl",
        f"/home/ubuntu/Graph-CoT/Graph-CoT/results/{args.model}/amazon/results_rephrased.jsonl",
    ]
    OUTPUT_FILE = f"/home/ubuntu/PolyG/examples/results/amazon/{args.model}/judgements_rephrased_sp*_nested.jsonl"
elif args.dataset == "goodreads":
    ANSWER_PATH = [
        f"/home/ubuntu/PolyG/examples/results/goodreads/{args.model}/results_rephrased.jsonl",
        f"/home/ubuntu/fast-graphrag/examples/results/goodreads/{args.model}/results_rephrased.jsonl",
        f"/home/ubuntu/Graph-CoT/Graph-CoT/results/{args.model}/goodreads/results_rephrased.jsonl",
    ]
    OUTPUT_FILE = f"/home/ubuntu/PolyG/examples/results/goodreads/{args.model}/judgements_rephrased.jsonl"
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

SYSTEM_ROLE = """
---Role---
You are an expert tasked with evaluating responses to the some questions based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
"""

PROMPT_WITH_GT = """
You will evaluate multiple responses to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question? A comprehensive answer should be thorough and complete, without being redundant or irrelevant. For example, if the question is ’What are the benefits and drawbacks of nuclear energy?’, a comprehensive answer would provide both the positive and negative aspects of nuclear energy, such as its efficiency, environmental impact, safety, cost, etc. A comprehensive answer should not leave out any important points or provide irrelevant information. For example, an incomplete answer would only provide the benefits of nuclear energy without describing the drawbacks, or a redundant answer would repeat the same information multiple times.
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question? A diverse answer should be multi-faceted and multi-dimensional, offering different viewpoints and angles on the question. For example, if the question is ’What are the causes and effects of climate change?’, a diverse answer would provide different causes and effects of climate change, such as greenhouse gas emissions, deforestation, natural disasters, biodiversity loss, etc. A diverse answer should also provide different sources and evidence to support the answer. For example, a single-source answer would only cite one source or evidence, or a biased answer would only provide one perspective or opinion.
- **Empowerment**: How well does the answer help the reader understand and make informed judgements about the topic without being misled or making fallacious assumptions? Evaluate each answer on the quality of answer as it relates to clearly explaining and providing reasoning and sources behind the claims in the answer.
- **Directness**: How specifically and clearly does the answer address the question? A direct answer should provide a clear and concise answer to the question. For example, if the question is ’What is the capital of France?’, a direct answer would be ’Paris’. A direct answer should not provide any irrelevant or unnecessary information that does not answer the question. For example, an indirect answer would be ’The capital of France is located on the river Seine’.

For each criterion, choose the best response(s) and explain the reason for this decision.
Then, select the overall winner(s) by jointly consider the above four criteria and their importance to the question. For example, if the question is about a specific topic, the overall winner should prefer the response that is most relevant and directly answers the question. If the question is more open-ended, the overall winner should prefer the response that is most comprehensive and diverse in its coverage of the topic.

Note:
1. You will be provided with the ground-truth answers to each question, and you should evaluate the reponses based on the groun-truth answers. Good reponses should be consistent with the ground-truth answers.
2. There can be multiple winners for a question in the case where they all well answer the question regarding the criteria. You also need to give the reasons for this case.
3. Reponse like "there is no direct information for me to answer" or other forms that indicat it can not give answers to the question is not a valid answer as all questions are designed to ensure there is an answer. These kinds of reponses should be considered as a bad reponse under all three criteria.
4. In some cases that one response could be diverse and informative about some other knowledge but is off-topic and irrelevant to the question, that response should be considered as a bad answer. Only responses that are actually helpful to answer the question should be considered valid responses, otherwise the response is bad under all criteria.

Question:
{query}

Ground Truth Answers:
{gt_answer}

Responses:
{answer}

Evaluate the above responses using the three criteria and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format (wrap the JSON in triple backticks):

```json
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
    "Directness": {{
        "Winner": "[Method name 1], [Method name 2], ...",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Method name 1], [Method name 2], ...",
        "Explanation": "[Summarize why this answer is the overall winner based on all criteria]"
    }}
}}
```
"""

PROMPT_WITHOUT_GT = """
You will evaluate multiple responses to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question? A comprehensive answer should be thorough and complete, without being redundant or irrelevant. For example, if the question is ’What are the benefits and drawbacks of nuclear energy?’, a comprehensive answer would provide both the positive and negative aspects of nuclear energy, such as its efficiency, environmental impact, safety, cost, etc. A comprehensive answer should not leave out any important points or provide irrelevant information. For example, an incomplete answer would only provide the benefits of nuclear energy without describing the drawbacks, or a redundant answer would repeat the same information multiple times.
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question? A diverse answer should be multi-faceted and multi-dimensional, offering different viewpoints and angles on the question. For example, if the question is ’What are the causes and effects of climate change?’, a diverse answer would provide different causes and effects of climate change, such as greenhouse gas emissions, deforestation, natural disasters, biodiversity loss, etc. A diverse answer should also provide different sources and evidence to support the answer. For example, a single-source answer would only cite one source or evidence, or a biased answer would only provide one perspective or opinion.
- **Empowerment**: How well does the answer help the reader understand and make informed judgements about the topic without being misled or making fallacious assumptions? Evaluate each answer on the quality of answer as it relates to clearly explaining and providing reasoning and sources behind the claims in the answer.
- **Directness**: How specifically and clearly does the answer address the question? A direct answer should provide a clear and concise answer to the question. For example, if the question is ’What is the capital of France?’, a direct answer would be ’Paris’. A direct answer should not provide any irrelevant or unnecessary information that does not answer the question. For example, an indirect answer would be ’The capital of France is located on the river Seine’.

For each criterion, choose the best response(s) and explain the reason for this decision.
Then, select the overall winner(s) by jointly consider the above four criteria and their importance to the question. For example, if the question is about a specific topic, the overall winner should prefer the response that is most relevant and directly answers the question. If the question is more open-ended, the overall winner should prefer the response that is most comprehensive and diverse in its coverage of the topic.

Note:
1. There can be multiple winners for a question in the case where they all well answer the question regarding the criteria. You also need to give the reasons for this case.
2. Reponse like "there is no direct information for me to answer" or other forms that indicat it can not give answers to the question is not a valid answer as all questions are designed to ensure there is an answer. These kinds of reponses should be considered as a bad reponse under all three criteria..
3. In some cases that one response could be diverse and informative about some other knowledge but is off-topic and irrelevant to the question, that response should be considered as a bad answer. Only responses that are actually helpful to answer the question should be considered valid responses, otherwise the response is bad under all criteria.

Question:
{query}

Responses:
{answer}

Evaluate the above responses using the three criteria and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format (wrap the JSON in triple backticks):

```json
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
    "Directness": {{
        "Winner": "[Method name 1], [Method name 2], ...",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Method name 1], [Method name 2], ...",
        "Explanation": "[Summarize why this answer is the overall winner based on all criteria]"
    }}
}}
```
"""

ERROR_MSG = "When processing your generated json evaluation result, errors occurred which indicates that you have made a mistake. Please fix the error and generate the response again. The error is: {}."


def bedrock_generator(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: List[Tuple] = [],
    **kwargs,
) -> str:
    messages, system = [], []
    if system_prompt:
        system.append({"text": system_prompt})

    history_messages_formated = [
        {"role": r, "content": [{"text": m}]} for r, m in history_messages
    ]

    messages.extend(history_messages_formated)
    messages.append({"role": "user", "content": [{"text": prompt}]})

    response = bedrock_cli.converse(modelId=MODEL_ID, messages=messages, system=system)
    return response["output"]["message"]["content"][0]["text"]


def openai_generator(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: List[dict] = [],
    **kwargs,
) -> str | None:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    history_messages = [{"role": r, "content": m} for r, m in history_messages]

    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=args.model, messages=messages, stream=False
    )
    return response.choices[0].message.content


answers = []
for path in ANSWER_PATH:
    with open(path, "r") as f:
        for item in jsonlines.Reader(f):
            answers.append(item)

question_types = [
    # "single_entity_abstract_rephrased",
    "single_entity_concrete_rephrased",
    # "multi_entity_abstract_rephrased",
    # "multi_entity_concrete_rephrased",
    "nested_question_rephrased",
]
question_answer = {key: defaultdict(list) for key in question_types}
for item in answers:
    if item["question_type"] not in question_answer.keys():
        continue
    question_answer[item["question_type"]][item["question"]].append(
        (item["method"], item["model_answer"], item["gt_answer"])
    )

for question_type, qa_pairs in question_answer.items():
    print(f"Total number of questions: {len(qa_pairs)}")

criteria = [
    "Comprehensiveness",
    "Diversity",
    "Empowerment",
    "Directness",
    "Overall Winner",
]
method_names = [
    "BFS",
    "cypher_single_entity",
    "cypher_only",
    "Fastgraphrag_PPR",
    "GraphCoT",
    "adaptive",
]
all_method_wins = {}
for question_type in question_types:
    method_wins = {name: {method: 0 for method in method_names} for name in criteria}
    for it, (question, answers) in enumerate(question_answer[question_type].items()):

        question = question.replace('"', "'")
        print(f"Question {it + 1}: {question}")
        gt, prompt, result = "N/A", "", ""

        answer_str = "Answers:\n\n"
        for it, answer_tuple in enumerate(answers):
            method, answer, gt = answer_tuple
            answer_str += "-------------------------------------\n"
            answer_str += f"Answer {it + 1} (Method {method}): {answer}\n\n"
        answer_str += "-------------------------------------\n"

        sys_prompt = PROMPT_WITH_GT if gt != "N/A" else PROMPT_WITHOUT_GT
        history_msgs = []

        while True:
            try:
                prompt = sys_prompt.format(
                    query=question,
                    answer=answer_str,
                    question_type=question_type,
                    gt_answer=gt,
                )
                result = bedrock_generator(
                    prompt=prompt,
                    system_prompt=SYSTEM_ROLE,
                    history_messages=history_msgs,
                )
                assert result is not None, "No response from the model."
                print(result)
                result = result.split("```")[1].strip("json")

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

                break
            except Exception as e:
                print(f"Error: {e}")
                history_msgs.extend(
                    [
                        ("user", prompt),
                        ("assistant", result),
                        ("user", ERROR_MSG.format(str(e))),
                    ]
                )

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
