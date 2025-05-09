import jsonlines
from collections import defaultdict


# ANSWER_PATH = [
#     "/home/ubuntu/PolyG/examples/results/Physics/claude-3.5-sonnet/results_rephrased.jsonl",
#     "/home/ubuntu/fast-graphrag/examples/results/Physics/results_rephrased.jsonl",
#     "/home/ubuntu/Graph-CoT/Graph-CoT/results/claude-3-5-sonnet/maple-Physics/results_rephrased.jsonl",
# ]

# ANSWER_PATH = [
#     "/home/ubuntu/PolyG/examples/results/goodreads/claude-3.5-sonnet/results_rephrased.jsonl",
#     "/home/ubuntu/fast-graphrag/examples/results/goodreads/results_rephrased.jsonl",
#     "/home/ubuntu/Graph-CoT/Graph-CoT/results/claude-3-5-sonnet/goodreads/results_rephrased.jsonl",
# ]

ANSWER_PATH = [
    "/home/ubuntu/PolyG/examples/results/amazon/claude-3.5-sonnet/results_rephrased.jsonl",
    "/home/ubuntu/fast-graphrag/examples/results/amazon/results_rephrased.jsonl",
    "/home/ubuntu/Graph-CoT/Graph-CoT/results/claude-3-5-sonnet/amazon/results_rephrased.jsonl",
]


answers = []
for path in ANSWER_PATH:
    with open(path, "r") as f:
        for item in jsonlines.Reader(f):
            answers.append(item)

question_types = [
    "single_entity_abstract_rephrased",
    "single_entity_concrete_rephrased",
    "multi_entity_abstract_rephrased",
    "multi_entity_concrete_rephrased",
    "nested_question_rephrased",
]
question_answer = {key: defaultdict(list) for key in question_types}
for item in answers:
    question_answer[item["question_type"]][item["question"]].append(item)

method_names = [
    "BFS",
    "cypher_single_entity",
    "Fastgraphrag_PPR",
    "GraphCoT",
    # "shortest_paths",
    # "cypher_multi_entity",
    "cypher_only",
    "adaptive",
]
all_time, all_tokens, all_api_calls = [], [], []
bfs_time = []
for question_type in question_types:
    method_time = {method: 0 for method in method_names}
    method_tokens = {method: 0 for method in method_names}
    method_api_calss = {method: 0 for method in method_names}
    method_counts = {method: 0 for method in method_names}
    for it, (question, answers) in enumerate(question_answer[question_type].items()):
        for it, answer in enumerate(answers):
            method = answer["method"]
            method_counts[method] += 1
            method_time[method] += float(answer["duration"])
            method_tokens[method] += int(answer["token_count"])
            method_api_calss[method] += int(answer["api_calls"])

            if method == "BFS":
                bfs_time.append(float(answer["duration"]))

    for method, duration in method_time.items():
        if method_counts[method] != 0:
            method_time[method] = round(duration / method_counts[method], 2)

    for method, tokens in method_tokens.items():
        if method_counts[method] != 0:
            method_tokens[method] = int(tokens / method_counts[method])

    for method, api_calls in method_api_calss.items():
        if method_counts[method] != 0:
            method_api_calss[method] = int(api_calls / method_counts[method])

    print("=" * 80)
    print(f"Question Type: {question_type}")
    print(method_time)
    print(method_tokens)
    print(method_api_calss)

    # print it as csv format
    print(",".join(method_names))
    print("=" * 80)
    print(",".join([str(method_time[method]) for method in method_names]))
    print("=" * 80)
    print(",".join([str(method_tokens[method]) for method in method_names]))
    print("=" * 80)
    print(",".join([str(method_api_calss[method]) for method in method_names]))

    all_time.append(method_time)
    all_tokens.append(method_tokens)
    all_api_calls.append(method_api_calss)

print("=" * 80)
print("All Time")
print(",".join(method_names))
for method_time in all_time:
    print(",".join([str(method_time[method]) for method in method_names]))
print("Overall Times")
print(",".join(method_names))
overall_time = {method: 0 for method in method_names}
for method in method_names:
    for method_time in all_time:
        overall_time[method] += method_time[method]
    overall_time[method] = round(overall_time[method] / len(all_time), 2)
print(",".join([str(overall_time[method]) for method in method_names]))

print("=" * 80)
print("All Tokens")
print(",".join(method_names))
for method_tokens in all_tokens:
    print(",".join([str(method_tokens[method]) for method in method_names]))
print("Overall Tokens")
print(",".join(method_names))
overall_tokens = {method: 0 for method in method_names}
for method in method_names:
    for method_tokens in all_tokens:
        overall_tokens[method] += method_tokens[method]
    overall_tokens[method] = int(overall_tokens[method] / len(all_tokens))
print(",".join([str(overall_tokens[method]) for method in method_names]))

# print("=" * 80)
# print("All API Calls")
# print(",".join(method_names))
# for method_api_calss in all_api_calls:
#     print(",".join([str(method_api_calss[method]) for method in method_names]))
