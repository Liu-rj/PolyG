import numpy as np
import jsonlines
from collections import defaultdict


ANSWER_PATH = [
    "/home/ubuntu/PolyG/examples/results/Physics/claude-3.5-sonnet/results_rephrased_final.jsonl",
    "/home/ubuntu/PolyG/examples/results/goodreads/claude-3.5-sonnet/results_rephrased_final.jsonl",
    "/home/ubuntu/PolyG/examples/results/amazon/claude-3.5-sonnet/results_rephrased_final.jsonl",
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
question_answer = {key: [] for key in question_types}
for item in answers:
    if item["method"] != "adaptive":
        continue
    question_answer[item["question_type"]].append(item)

total_questions = 0
for question_type, qa_pairs in question_answer.items():
    total_questions += len(qa_pairs)
    print(f"Number of questions: {len(qa_pairs)}")
print(f"Total number of questions: {total_questions}")

all_acc = []
for i, question_type in enumerate(question_types):
    correct = 0
    gt = i if i <= 3 else -1
    for it, item in enumerate(question_answer[question_type]):
        cls = int(item["question_classification_result"])
        if cls == gt:
            correct += 1
        if question_type == "multi_entity_concrete_rephrased" and cls == 1:
            correct += 1
    print(
        f"Accuracy for {question_type}: {correct / len(question_answer[question_type])}"
    )
    all_acc.append(correct / len(question_answer[question_type]))
print(f"Overall accuracy: {np.mean(all_acc)}")

# plot the accuracy
import matplotlib.pyplot as plt

FONT_SIZE = 24
x_labels = ["<s,*,*>", "<s,p,*>", "<s,*,o>", "<s,p,o>", "Nested"]

plt.figure(figsize=(10, 6))
plt.bar(x_labels, all_acc, color="skyblue", width=0.45, edgecolor="k")
# plot the average accuracy
plt.axhline(y=sum(all_acc) / len(all_acc), color="g", linestyle="--", label="Average")
# write the average accuracy on the plot
plt.text(
    3.88,
    np.mean(all_acc) + 0.02,
    f"Average: {np.mean(all_acc):.2f}",
    ha="center",
    fontsize=FONT_SIZE - 4,
)
plt.xlabel("Question Pattern", fontsize=FONT_SIZE)
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.ylabel("Classification Accuracy", fontsize=FONT_SIZE)
# plt.title("Accuracy for Different Question Types")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("qc_accuracy.pdf", bbox_inches="tight")
