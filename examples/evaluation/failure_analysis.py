import numpy as np
import jsonlines
from collections import defaultdict
from polyg.prompt import PROMPTS

# ANSWER_PATH = [
#     "/home/renjie/PolyG/examples/results/physics/claude-3.5-sonnet/results_rephrased_final.jsonl",
#     "/home/renjie/PolyG/examples/results/goodreads/claude-3.5-sonnet/results_rephrased_final.jsonl",
#     "/home/renjie/PolyG/examples/results/amazon/claude-3.5-sonnet/results_rephrased_final.jsonl",
# ]

# ANSWER_PATH = [
#     "/home/renjie/PolyG/examples/results/physics/Qwen/Qwen3-14B/results.jsonl",
#     "/home/renjie/PolyG/examples/results/goodreads/Qwen/Qwen3-14B/results.jsonl",
#     "/home/renjie/PolyG/examples/results/amazon/Qwen/Qwen3-14B/results.jsonl",
# ]

ANSWER_PATH = [
    "/home/renjie/PolyG/examples/results/physics/deepseek-r1/results_rephrased.jsonl",
    "/home/renjie/PolyG/examples/results/goodreads/deepseek-r1/results_rephrased.jsonl",
    "/home/renjie/PolyG/examples/results/amazon/deepseek-r1/results_rephrased.jsonl",
]

answers = []
for path in ANSWER_PATH:
    with open(path, "r") as f:
        for item in jsonlines.Reader(f):
            answers.append(item)
print(f"Number of answers: {len(answers)}")

question_types = [
    "single_entity_abstract_rephrased",
    "single_entity_concrete_rephrased",
    "multi_entity_abstract_rephrased",
    "multi_entity_concrete_rephrased",
    "nested_question_rephrased",
]
question_answer = {key: defaultdict(list) for key in question_types}
for item in answers:
    if item["question_type"] not in question_answer.keys():
        continue
    question_answer[item["question_type"]][item["method"]].append(item)

total_questions = 0
for question_type, qa_pairs in question_answer.items():
    total_questions += len(qa_pairs["adaptive"])
    print(f"Number of questions: {len(qa_pairs["adaptive"])}")
print(f"Total number of questions: {total_questions}")

method_names = [
    "cypher_single_entity",
    "cypher_only",
    "adaptive",
]
method_failures = [[] for _ in range(len(method_names))]
for question_type, qa_pairs in question_answer.items():
    print("+" * 50)
    for i, method in enumerate(method_names):
        items = qa_pairs[method]
        failures = 0
        for it, item in enumerate(items):
            if (
                item["model_answer"] == PROMPTS["fail_response"]
                or item["model_answer"] == PROMPTS["token_limit_exceeded"]
            ):
                failures += 1
        failure_rate = round(failures / len(items), 4)
        method_failures[i].append(failure_rate)
        print(f"Failure rate for {question_type} with method {method}: {failure_rate}")
    print("-" * 50)

for i, method in enumerate(method_names):
    print(f"Average failure rate for {method}: {round(np.mean(method_failures[i]), 4)}")
    print(f"Standard deviation for {method}: {round(np.std(method_failures[i]), 4)}")
    print("-" * 50)

# plot the failure rates
import matplotlib.pyplot as plt

FONT_SIZE = 24
x_labels = ["<s,*,*>", "<s,p,*>", "<s,*,o>", "<s,p,o>", "Nested"]
methods = ["RoG", "Cypher", "PolyG"]
colors = ["silver", "bisque", "#BDD7EE"]
fig, ax = plt.subplots(figsize=(10, 6))

num_groups = len(x_labels)
x = np.arange(num_groups)  # group positions

bar_width = 0.25
offsets = [-bar_width, 0, bar_width]  # position offsets for each method

# Plot bars for each method
for i, method_data in enumerate(method_failures):
    ax.bar(
        x + offsets[i],
        method_data,
        width=bar_width,
        label=methods[i],
        color=colors[i],
        edgecolor="k",
    )

# Formatting
ax.set_xlabel("Question Pattern", fontsize=FONT_SIZE)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=FONT_SIZE, rotation=45)
plt.yticks(np.arange(0, 1, 0.2), fontsize=FONT_SIZE)
ax.set_ylabel("Failure Rate", fontsize=FONT_SIZE)
ax.legend(methods, fontsize=FONT_SIZE - 2, loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("failure_rate_deepseek.pdf", bbox_inches="tight")
