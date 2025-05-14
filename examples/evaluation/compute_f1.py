import boto3
import jsonlines
import argparse
from collections import defaultdict


argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="physics", required=True)
args = argparser.parse_args()


if args.dataset == "physics":
    ANSWER_PATH = [
        "/home/ubuntu/PolyG/examples/results/Physics/claude-3.5-sonnet/results_rephrased_final.jsonl",
        "/home/ubuntu/fast-graphrag/examples/results/Physics/claude-3.5-sonnet/results_rephrased_final.jsonl",
        "/home/ubuntu/Graph-CoT/Graph-CoT/results/claude-3-5-sonnet/maple-Physics/results_rephrased_final.jsonl",
    ]
elif args.dataset == "amazon":
    ANSWER_PATH = [
        "/home/ubuntu/PolyG/examples/results/amazon/claude-3.5-sonnet/results_rephrased_final.jsonl",
        "/home/ubuntu/fast-graphrag/examples/results/amazon/claude-3.5-sonnet/results_rephrased_final.jsonl",
        "/home/ubuntu/Graph-CoT/Graph-CoT/results/claude-3-5-sonnet/amazon/results_rephrased_final.jsonl",
    ]
elif args.dataset == "goodreads":
    ANSWER_PATH = [
        "/home/ubuntu/PolyG/examples/results/goodreads/claude-3.5-sonnet/results_rephrased_final.jsonl",
        "/home/ubuntu/fast-graphrag/examples/results/goodreads/claude-3.5-sonnet/results_rephrased_final.jsonl",
        "/home/ubuntu/Graph-CoT/Graph-CoT/results/claude-3-5-sonnet/goodreads/results_rephrased_final.jsonl",
    ]


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

method_names = [
    # "BFS",
    "cypher_single_entity",
    # "Fastgraphrag_PPR",
    # "GraphCoT",
    # "cypher_only",
    # "adaptive",
]
question_answer = defaultdict(list)
for item in answers:
    if item["gt_answer"] == "N/A" or item["method"] not in method_names:
        continue
    if item["question_type"] != "nested_question_rephrased":
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
        assert answer["question_type"] in [
            "single_entity_concrete_rephrased",
            "nested_question_rephrased",
        ]
        method, gt = answer["method"], answer["gt_answer"]
        answer = (
            answer["answer_list"]
            if "answer_list" in answer and answer["answer_list"] != "N/A"
            else answer["model_answer"]
        )
        # answer = answer["model_answer"]

        result = bedrock_generator(
            prompt=PROMPT.format(query=question, reponse=answer),
            system_prompt=SYSTEM_ROLE,
        )
        result = result.strip("[]").split(", ")

        for i in range(len(result)):
            result[i] = result[i].strip('"').lower()

        for i in range(len(gt)):
            gt[i] = gt[i].strip('"').lower()

        tp = len(set(result) & set(gt))
        fp = len(set(result) - set(gt))
        fn = len(set(gt) - set(result))
        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall != 0
            else 0
        )
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
