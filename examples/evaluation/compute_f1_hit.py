import asyncio
import os
import jsonlines
import argparse
import string
import re
from collections import defaultdict
from openai import AsyncOpenAI


argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="physics", required=True)
argparser.add_argument("--model", type=str, default="claude-3.5-sonnet", required=True)
args = argparser.parse_args()


ANSWER_PATH = [
    f"{os.getenv('HOME')}/fast-graphrag/examples/results/{args.dataset}/{args.model}/results.jsonl",
    f"{os.getenv('HOME')}/Graph-CoT/Graph-CoT/results/{args.model}/{args.dataset}/results.jsonl",
    f"{os.getenv('HOME')}/PolyG/examples/results/{args.dataset}/{args.model}/results.jsonl",
]
OUTPUT_PATH = f"{os.getenv('HOME')}/PolyG/examples/results/{args.dataset}/{args.model}/detailed_evaluation.jsonl"


client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
)

SYSTEM_ROLE = """
---Role---
You are an expert tasked with extracting answers in user responses and output in specific format.
"""

PROMPT = """
You will be provided with a question and a reponse from a user.

You need to extract the exact answers from the user reponse and output them in the list format.

Example 1:
Question: "Who are the authors of the paper 'self trapping of optical beams in photorefractive media'?"
User Response: "Based on the information provided in the relationships table, the authors of the paper \"Self trapping of optical beams in photorefractive media\" are:\n\nMordechai Segev\nBruno Crosignani\nAmnon Yariv\nDoruk Engin\nPaolo Di Porto\nGregory J. Salamo\n\nThis can be inferred from the multiple relationships listing these individuals as authors of the paper in question."
Output: ["Mordechai Segev", "Bruno Crosignani", "Amnon Yariv", "Doruk Engin", "Paolo Di Porto", "Gregory J. Salamo"]

Example 2:
Question: "who is michael buble?"
User Response: "Michael Bublé is a Canadian singer and songwriter."
Output: ["singer", "songwriter"]

Example 3:
Question: "what did peter tchaikovsky do"
User Response: "Peter Tchaikovsky was a composer. He composed numerous famous works including:\n\n1. Ballets like \"Swan Lake\", \"The Nutcracker\", and \"The Sleeping Beauty\"\n2. Operas such as \"Eugene Onegin\" and \"The Queen of Spades\"\n3. Symphonies, including his Symphony No. 4, No. 5, and No. 6 (\"Pathetique\")\n4. Concertos, like the Piano Concerto No. 1 and the Violin Concerto\n5. Other orchestral works such as the \"1812 Overture\" and \"Romeo and Juliet\"\n\nTchaikovsky was particularly renowned for his ballet music and is considered one of the greatest composers of the Romantic era."
Output: ["Composer", "Musician", "Librettist"]

Note:
1. Keep the answer phrases as they are in the reponses and do not change them in any way.
2. If the reponse does not contain any answer, output an empty list: [].

---Question and Reponse---

Question:
{query}

Response:
{reponse}

Your Output Format (exactly as shown below, encapsulated in triple backticks):
```
["Answer 1", "Answer 2", ...]
```
"""


async def openai_generator(
    prompt: str,
    system_prompt: str | None = None,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(
        model="deepseek-chat", messages=messages, stream=False
    )
    return response.choices[0].message.content  # type: ignore


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_acc(prediction, answer):
    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0


def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall


async def evaluate_question_responses(question, answers):
    print(f"Question: {question}, Number of answers: {len(answers)}")
    answers_score = []
    for answer in answers:
        assert answer["question_type"] in ["object_discovery", "nested_question"]

        method, gt = answer["method"], answer["gt_answer"]
        for i in range(len(gt)):
            gt[i] = gt[i].strip('"').lower()

        if "answer_list" in answer and len(answer["answer_list"]) > 0:
            result = answer["answer_list"]
            for i in range(len(result)):
                result[i] = result[i].strip('"').lower()
        else:
            response = await openai_generator(
                prompt=PROMPT.format(query=question, reponse=answer["model_answer"]),
                system_prompt=SYSTEM_ROLE,
            )

            print(response)
            result_str = response.split("```")[1]
            try:
                result = eval(result_str)
            except Exception as e:
                print(e)
                result = result_str.strip("[]").split(", ")
            for i in range(len(result)):
                result[i] = str(result[i]).strip('"').lower()

        # precision, recall, f1 = compute_score(result, gt)
        f1, precision, recall = eval_f1(result, gt)
        prediction_str = " ".join(result)
        acc = eval_acc(prediction_str, gt)
        hit = eval_hit(prediction_str, gt)

        answers_score.append(
            {
                "question_type": answer["question_type"],
                "question": question,
                "method": method,
                "answer": result,
                "ground_truth": gt,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "acc": acc,
                "hit": hit,
            }
        )

        print(answers_score)

    return answers_score


async def main_evaluation(question_answer_pairs):
    return await asyncio.gather(
        *[
            evaluate_question_responses(question, answers)
            for question, answers in question_answer_pairs.items()
        ]
    )


answers = []
for path in ANSWER_PATH:
    with open(path, "r") as f:
        for item in jsonlines.Reader(f):
            answers.append(item)

method_names = [
    "BFS",
    "cypher_single_entity",
    "Fastgraphrag_PPR",
    "GraphCoT",
    "cypher_only",
    "BFS+PPR",
    "adaptive",
]
question_answer = defaultdict(list)
for item in answers:
    if item["gt_answer"] == "N/A" or item["method"] not in method_names:
        continue
    question_answer[item["question"]].append(item)

print(f"Number of questions: {len(question_answer)}")

method_precision = {method: 0.0 for method in method_names}
method_recall = {method: 0.0 for method in method_names}
method_f1 = {method: 0.0 for method in method_names}
method_acc = {method: 0.0 for method in method_names}
method_hit = {method: 0.0 for method in method_names}
method_counts = {method: 0 for method in method_names}

all_records = asyncio.run(main_evaluation(question_answer))

for qa_records in all_records:
    for method_record in qa_records:
        method = method_record["method"]
        precision = method_record["precision"]
        recall = method_record["recall"]
        f1 = method_record["f1"]
        acc = method_record["acc"]
        hit = method_record["hit"]

        method_counts[method] += 1
        method_precision[method] += precision
        method_recall[method] += recall
        method_f1[method] += f1
        method_acc[method] += acc
        method_hit[method] += hit

        with jsonlines.open(OUTPUT_PATH, "a") as f:
            f.write(method_record)

for method in method_names:
    counts = method_counts[method]
    if counts != 0:
        method_precision[method] = round(method_precision[method] / counts, 4)
        method_recall[method] = round(method_recall[method] / counts, 4)
        method_f1[method] = round(method_f1[method] / counts, 4)
        method_acc[method] = round(method_acc[method] / counts, 4)
        method_hit[method] = round(method_hit[method] / counts, 4)

print("=" * 80)
print(method_f1)
print(",".join(method_names))
print(",".join([str(method_precision[method]) for method in method_names]))
print(",".join([str(method_recall[method]) for method in method_names]))
print(",".join([str(method_f1[method]) for method in method_names]))
print(",".join([str(method_acc[method]) for method in method_names]))
print(",".join([str(method_hit[method]) for method in method_names]))
