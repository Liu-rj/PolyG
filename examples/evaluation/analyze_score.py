import jsonlines
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="physics", required=True)
argparser.add_argument("--model", type=str, default="claude-3.5-sonnet", required=True)
args = argparser.parse_args()

RESULT_PATH = f"../results/{args.dataset}/{args.model}/detailed_evaluation.jsonl"

method_names = [
    "BFS",
    "cypher_single_entity",
    "Fastgraphrag_PPR",
    "GraphCoT",
    "cypher_only",
    "adaptive",
]

method_precision = {method: 0.0 for method in method_names}
method_recall = {method: 0.0 for method in method_names}
method_f1 = {method: 0.0 for method in method_names}
method_acc = {method: 0.0 for method in method_names}
method_hit = {method: 0.0 for method in method_names}
method_counts = {method: 0 for method in method_names}

with jsonlines.open(RESULT_PATH, "r") as reader:
    results = list(reader)

for item in results:
    method = item["method"]
    method_precision[method] += item["precision"]
    method_recall[method] += item["recall"]
    method_f1[method] += item["f1"]
    method_acc[method] += item["acc"]
    method_hit[method] += item["hit"]
    method_counts[method] += 1

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
print("Method," + ",".join(method_names))
print(
    "Precision," + ",".join([str(method_precision[method]) for method in method_names])
)
print("Recall," + ",".join([str(method_recall[method]) for method in method_names]))
print("F1," + ",".join([str(method_f1[method]) for method in method_names]))
print("Accuracy," + ",".join([str(method_acc[method]) for method in method_names]))
print("Hit," + ",".join([str(method_hit[method]) for method in method_names]))
