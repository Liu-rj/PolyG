import jsonlines
import argparse
import string
import re
from collections import defaultdict
from copy import deepcopy


argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="webqsp", required=True)
argparser.add_argument("--model", type=str, default="Qwen/Qwen3-14B", required=True)
args = argparser.parse_args()


ANSWER_PATH = [
    # f"{os.getenv('HOME')}/fast-graphrag/examples/results/{args.dataset}/{args.model}/results.jsonl",
    # f"{os.getenv('HOME')}/Graph-CoT/Graph-CoT/results/{args.model}/{args.dataset}/results.jsonl",
    f"results/{args.dataset}/{args.model}/results_subgraphrag.jsonl",
]
OUTPUT_PATH = (
    f"results/{args.dataset}/{args.model}/detailed_evaluation_subgraphrag.jsonl"
)


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
    if split is not None:
        return prediction.split(split)

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
    "subgraphrag",
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
for it, (question, answers) in enumerate(question_answer.items()):
    print(f"Question {it+1}: {question}, Number of answers: {len(answers)}")
    for answer in answers:
        method, gt = answer["method"], answer["gt_answer"]
        for i in range(len(gt)):
            gt[i] = gt[i].strip('"').lower()

        response = answer["model_answer"]
        print(response)
        result = get_pred(response, split=None)
        for i in range(len(result)):
            result[i] = str(result[i]).strip('"').lower()

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

        precision = eval_precision(result, gt, double_check)[0]
        recall = eval_recall(result, gt, double_check)[0]
        f1 = eval_f1(precision, recall)
        prediction_str = " ".join(result)
        acc = eval_acc(prediction_str, gt)
        hit = eval_hit(prediction_str, gt)

        method_counts[method] += 1
        method_precision[method] += precision
        method_recall[method] += recall
        method_f1[method] += f1
        method_acc[method] += acc
        method_hit[method] += hit

        result_entry = {
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
        print(result_entry)

        with jsonlines.open(OUTPUT_PATH, "a") as f:
            f.write(result_entry)

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
