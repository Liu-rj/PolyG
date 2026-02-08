import jsonlines
import argparse
from collections import defaultdict


argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="physics", required=True)
argparser.add_argument("--model", type=str, default="claude-3.5-sonnet", required=False)
args = argparser.parse_args()

ANSWER_PATH = [
    # f"results/{args.dataset}/{args.model}/results.jsonl",
    f"../subgraphrag/results/{args.dataset}/{args.model}/results_dc_100.jsonl",
    # f"/home/renjie/fast-graphrag/examples/results/{args.dataset}/{args.model}/results.jsonl",
    # f"/home/renjie/Graph-CoT/Graph-CoT/results/{args.model}/{args.dataset}/results.jsonl",
]

answers = []
for path in ANSWER_PATH:
    with open(path, "r") as f:
        for item in jsonlines.Reader(f):
            answers.append(item)

method_names = [
    "BFS",
    "Fastgraphrag_PPR",
    "GraphCoT",
    "cypher_only",
    "subgraphrag",
    "adaptive",
]
all_time, all_tokens, all_api_calls = [], [], []

method_time = {method: 0.0 for method in method_names}
method_tokens = {method: 0 for method in method_names}
method_api_calss = {method: 0 for method in method_names}
method_counts = {method: 0 for method in method_names}
for it, item in enumerate(answers):
    method = item["method"]
    method_counts[method] += 1
    method_time[method] += float(item["duration"])
    method_tokens[method] += int(item["token_count"])
    method_api_calss[method] += int(item["api_calls"])

for method, duration in method_time.items():
    if method_counts[method] != 0:
        method_time[method] = round(duration / method_counts[method], 2)

for method, tokens in method_tokens.items():
    if method_counts[method] != 0:
        method_tokens[method] = int(tokens / method_counts[method])

for method, api_calls in method_api_calss.items():
    if method_counts[method] != 0:
        method_api_calss[method] = int(api_calls / method_counts[method])

# print it as csv format
print("=" * 80)
print(",".join(method_names))
print(",".join([str(method_time[method]) for method in method_names]))
print(",".join([str(method_tokens[method]) for method in method_names]))
print(",".join([str(method_api_calss[method]) for method in method_names]))
