import jsonlines
import argparse
import string
import re
import os
from collections import defaultdict
from copy import deepcopy


argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="webqsp", required=True)
argparser.add_argument("--model", type=str, default="Qwen/Qwen3-14B", required=True)
args = argparser.parse_args()


ANSWER_PATH = f"{os.getenv('HOME')}/Graph-CoT/Graph-CoT/results/{args.model}/{args.dataset}/results.jsonl"
OUTPUT_PATH = f"results/{args.dataset}/{args.model}/detailed_evaluation.jsonl"


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


def eval_hit(prediction, answer, double_check):
    for a in answer:
        if "ans:" in prediction:
            all_pred = get_pred(prediction)
            for each_pred in all_pred:
                if match(each_pred, a):
                    return 1
                elif double_check and match(a, each_pred.split("ans:")[-1].strip()):
                    return 1
        else:
            if match(prediction, a):
                return 1
            elif double_check:
                all_pred = prediction.split("\n")
                for each_pred in all_pred:
                    if match(a, each_pred):
                        return 1
    return 0


def eval_recall(prediction, answer, double_check):
    prediction = deepcopy(prediction)
    prediction = sorted(prediction, key=len, reverse=True)
    matched = 0.0
    for a in answer:
        for pred in prediction:
            if match(pred, a):
                matched += 1
                prediction.remove(pred)
                break
            elif double_check:
                if match(a, pred.split("ans:")[-1].strip()) or match(a, pred):
                    matched += 1
                    prediction.remove(pred)
                    break
    return matched / len(answer), matched, len(answer)


def eval_precision(prediction, answer, double_check):
    prediction = deepcopy(prediction)
    prediction = sorted(prediction, key=len, reverse=True)
    num_pred = len(prediction)
    if num_pred == 0:
        return 0, 0, 0
    matched = 0.0
    for a in answer:
        for pred in prediction:
            if match(pred, a):
                matched += 1
                prediction.remove(pred)
                break
            elif double_check:
                if match(a, pred.split("ans:")[-1].strip()) or match(a, pred):
                    matched += 1
                    prediction.remove(pred)
                    break
    return matched / num_pred, matched, num_pred


def eval_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def get_pred(prediction, split=None):
    if isinstance(prediction, list):
        return remove_duplicates(prediction)
    if split is not None:
        return remove_duplicates(prediction.split(split))

    res = [p for p in prediction.split("\n") if "ans:" in p and "none" not in p.lower()]
    if len(res) >= 1:
        res = [
            p
            for p in res
            if "ans: not available" not in p.lower()
            and "ans: no information available" not in p.lower()
        ]
    return remove_duplicates(res)


answers = []
with open(ANSWER_PATH, "r") as f:
    for item in jsonlines.Reader(f):
        answers.append(item)

method_names = ["GraphCoT"]
question_answer = defaultdict(list)
for item in answers:
    if item["gt_answer"] == "N/A" or item["method"] not in method_names:
        continue
    question_answer[item["question"]].append(item)

print(f"Number of questions: {len(question_answer)}")

method_precision = {method: 0.0 for method in method_names}
method_recall = {method: 0.0 for method in method_names}
method_f1 = {method: 0.0 for method in method_names}
method_hit = {method: 0.0 for method in method_names}
method_counts = {method: 0 for method in method_names}
for it, (question, answers) in enumerate(question_answer.items()):
    print(f"Question {it+1}: {question}, Number of answers: {len(answers)}")
    for answer in answers:
        method, gt = answer["method"], answer["gt_answer"]

        gt = sorted(remove_duplicates(gt), key=len, reverse=True)
        if "when" in question.lower() or "what year" in question.lower():
            for idx in range(len(gt)):
                if "-" in gt[idx] and gt[idx].split("-")[0].isdigit():
                    gt[idx] = gt[idx].split("-")[0]

        double_check = any(
            [
                keyword in question.lower()
                for keyword in [
                    "when",
                    "what year",
                    "which year",
                    "where",
                    "sport",
                    "what countr",
                    "language",
                    "nba finals",
                    "world series",
                ]
            ]
        )

        response = str(answer["model_answer"])
        print(response)
        result = get_pred(response, split=",")

        precision = eval_precision(result, gt, double_check)[0]
        recall = eval_recall(result, gt, double_check)[0]
        f1 = eval_f1(precision, recall)
        prediction_str = " ".join(result)
        hit = eval_hit(prediction_str, gt, double_check)

        method_counts[method] += 1
        method_precision[method] += precision
        method_recall[method] += recall
        method_f1[method] += f1
        method_hit[method] += hit

        result_entry = {
            "question": question,
            "method": method,
            "answer": result,
            "ground_truth": gt,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "hit": hit,
        }
        print(result_entry)

        # with jsonlines.open(OUTPUT_PATH, "a") as f:
        #     f.write(result_entry)

for method in method_names:
    counts = method_counts[method]
    if counts != 0:
        method_precision[method] = round(method_precision[method] / counts, 4) * 100
        method_recall[method] = round(method_recall[method] / counts, 4) * 100
        method_f1[method] = round(method_f1[method] / counts, 4) * 100
        method_hit[method] = round(method_hit[method] / counts, 4) * 100

print(f"Number of questions: {len(question_answer)}")
print("=" * 80)
print(method_f1)
print("Method," + ",".join(method_names))
print(
    "Precision," + ",".join([str(method_precision[method]) for method in method_names])
)
print("Recall," + ",".join([str(method_recall[method]) for method in method_names]))
print("F1," + ",".join([str(method_f1[method]) for method in method_names]))
print("Hit," + ",".join([str(method_hit[method]) for method in method_names]))
