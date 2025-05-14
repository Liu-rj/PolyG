import jsonlines


# open results/WebQSP_result.jsonl
questions = []
with open("results/WebQSP_result.jsonl", "r") as f:
    for line in jsonlines.Reader(f):
        if line["response"].startswith("-1"):
            questions.append(line["question"])


