import jsonlines
from collections import defaultdict

# result_path = "results/WebQSP_result.jsonl"
result_path = "results/CWQ_result.jsonl"

# load WebQSP_result.jsonl
contents = []
with open(result_path, "r") as f:
    for item in jsonlines.Reader(f):
        contents.append(item)

print(f"Number of questions: {len(contents)}")

counts = defaultdict(int)
for content in contents:
    type = content["response"].split(":")[0].split("\n")[0]
    counts[type] += 1

print(counts)
