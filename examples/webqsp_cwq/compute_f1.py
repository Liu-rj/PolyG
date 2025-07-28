import boto3
import jsonlines
import argparse
from collections import defaultdict
from typing import List, Tuple


argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="physics", required=True)
argparser.add_argument("--model", type=str, default="claude-3.5-sonnet", required=True)
args = argparser.parse_args()


if args.dataset == "webqsp":
    ANSWER_PATH = [
        f"results/webqsp/{args.model}/results_rephrased.jsonl",
        # f"/home/ubuntu/fast-graphrag/examples/results/webqsp/{args.model}/results_rephrased.jsonl",
        # f"/home/ubuntu/Graph-CoT/Graph-CoT/results/{args.model}/maple-webqsp/results_rephrased.jsonl",
    ]
elif args.dataset == "cwq":
    ANSWER_PATH = [
        f"results/cwq/{args.model}/results_rephrased.jsonl",
        # f"/home/ubuntu/fast-graphrag/examples/results/cwq/{args.model}/results_rephrased.jsonl",
        # f"/home/ubuntu/Graph-CoT/Graph-CoT/results/{args.model}/cwq/results_rephrased.jsonl",
    ]
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")


CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
# CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
# CHAT_MODEL_ID = "us.deepseek.r1-v1:0"

SYSTEM_ROLE = """
---Role---
You are an expert tasked with extracting answers in user responses and output in specific format.
"""

PROMPT = """
You will be provided with a question and a reponse from a user.

You need to extract the exact answers from the user reponse and output them in the list format.

Example:
Question: "Who are the authors of the paper 'self trapping of optical beams in photorefractive media'?"

User Response: "Based on the information provided in the relationships table, the authors of the paper \"Self trapping of optical beams in photorefractive media\" are:\n\nMordechai Segev\nBruno Crosignani\nAmnon Yariv\nDoruk Engin\nPaolo Di Porto\nGregory J. Salamo\n\nThis can be inferred from the multiple relationships listing these individuals as authors of the paper in question."

Output: ["Mordechai Segev", "Bruno Crosignani", "Amnon Yariv", "Doruk Engin", "Paolo Di Porto", "Gregory J. Salamo"]

Note: Keep the answer phrases as they are in the reponses and do not change them in any way.

---Question and Reponse---

Question:
{query}

Response:
{reponse}

Output the extracted result in the the list format: ["Answer 1", "Answer 2", ...]
"""


def bedrock_generator(
    prompt: str,
    system_prompt: str | None = None,
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


def compute_score(result: List, gt: List) -> Tuple[float, float, float]:
    tp = len(set(result) & set(gt))
    fp = len(set(result) - set(gt))
    fn = len(set(gt) - set(result))
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return precision, recall, f1


answers = []
for path in ANSWER_PATH:
    with open(path, "r") as f:
        for item in jsonlines.Reader(f):
            answers.append(item)

method_names = [
    # "BFS",
    # "cypher_single_entity",
    # "Fastgraphrag_PPR",
    # "GraphCoT",
    # "cypher_only",
    "adaptive",
]
question_answer = defaultdict(list)
for item in answers:
    if item["gt_answer"] == "N/A" or item["method"] not in method_names:
        continue
    question_answer[item["question"]].append(item)

print(f"Number of questions: {len(question_answer)}")

method_precision = {method: 0 for method in method_names}
method_recall = {method: 0 for method in method_names}
method_f1 = {method: 0 for method in method_names}
method_counts = {method: 0 for method in method_names}
for it, (question, answers) in enumerate(question_answer.items()):
    print(f"Question {it+1}: {question}, Number of answers: {len(answers)}")
    for answer in answers:
        method, gt = answer["method"], answer["gt_answer"]
        for i in range(len(gt)):
            gt[i] = gt[i].strip('"').lower()

        result = answer["answer_list"].split(", ")
        for i in range(len(result)):
            result[i] = result[i].strip('"').lower()

        precision, recall, f1 = compute_score(result, gt)

        if f1 == 0:
            result = bedrock_generator(
                prompt=PROMPT.format(query=question, reponse=answer["model_answer"]),
                system_prompt=SYSTEM_ROLE,
            )

            result = result.strip("[]").split(", ")
            for i in range(len(result)):
                result[i] = result[i].strip('"').lower()

            precision, recall, f1 = compute_score(result, gt)

        print(f"Method: {method}, Precision: {precision}, Recall: {recall}, F1: {f1}")

        method_counts[method] += 1
        method_precision[method] += precision
        method_recall[method] += recall
        method_f1[method] += f1

for method in method_names:
    counts = method_counts[method]
    if counts != 0:
        method_precision[method] = round(method_precision[method] / counts, 4)
        method_recall[method] = round(method_recall[method] / counts, 4)
        method_f1[method] = round(method_f1[method] / counts, 4)

print("=" * 80)
print(method_f1)
print(",".join(method_names))
print(",".join([str(method_precision[method]) for method in method_names]))
print(",".join([str(method_recall[method]) for method in method_names]))
print(",".join([str(method_f1[method]) for method in method_names]))
