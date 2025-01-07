import jsonlines

answers = []
with open(
    "/home/ubuntu/graphrag_planner/examples/results/goodreads/results_archive.jsonl",
    "r",
) as f:
    for item in jsonlines.Reader(f):
        if item["method"] == "BFS":
            continue
        answers.append(item)

with jsonlines.open(
    "/home/ubuntu/graphrag_planner/examples/results/goodreads/results.jsonl", "w"
) as writer:
    for row in answers:
        writer.write(row)
