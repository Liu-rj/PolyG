import jsonlines
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="physics", required=True)
argparser.add_argument("--model", type=str, default="claude-3.5-sonnet", required=False)
args = argparser.parse_args()


if args.dataset == "physics":
    LLM_JUDGE_PATH = f"../results/physics/{args.model}/judgements_w_bfs_ppr.jsonl"
elif args.dataset == "amazon":
    LLM_JUDGE_PATH = f"../results/amazon/{args.model}/judgements.jsonl"
elif args.dataset == "goodreads":
    LLM_JUDGE_PATH = f"../results/goodreads/{args.model}/judgements.jsonl"
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")


judgements = []
with jsonlines.open(LLM_JUDGE_PATH, "r") as f:
    for item in f:
        judgements.append(item)

print(f"Number of judgements: {len(judgements)}")


question_types = [
    "subject_centered",
    "object_discovery",
    "predicate_discovery",
    "fact_check",
    "nested_question",
]
question_judgement = {key: [] for key in question_types}
for judgement in judgements:
    question_judgement[judgement["question_type"]].append(judgement)

criteria = [
    "Comprehensiveness",
    "Diversity",
    "Empowerment",
    "Directness",
    "Overall Winner",
]
method_names = [
    "BFS",
    "cypher_single_entity",
    "Fastgraphrag_PPR",
    "GraphCoT",
    "cypher_only",
    "BFS+PPR",
    "adaptive",
]
method_names = [name.lower() for name in method_names]
global_method_wins = {name: {method: 0 for method in method_names} for name in criteria}
all_questions = 0
for question_type in question_types:
    method_wins = {name: {method: 0.0 for method in method_names} for name in criteria}
    for judgement in question_judgement[question_type]:
        for criterion in criteria:
            winners = judgement[criterion]["Winner"]
            winners = winners if isinstance(winners, list) else winners.split(", ")
            for winner in winners:
                winner = winner.lower().lstrip("method").strip()
                if winner in method_wins[criterion]:
                    method_wins[criterion][winner] += 1
                    global_method_wins[criterion][winner] += 1

    all_questions += len(question_judgement[question_type])
    for criterion, methods in method_wins.items():
        for method, value in methods.items():
            if len(question_judgement[question_type]) > 0:
                method_wins[criterion][method] = round(
                    value / len(question_judgement[question_type]), 4
                )
            else:
                method_wins[criterion][method] = 0

    print("=" * 80)
    print(f"Question Type: {question_type}")
    print(method_wins)

    # print it as csv format
    print(",".join(method_names))
    for criterion in criteria:
        print(
            ",".join([str(method_wins[criterion][method]) for method in method_names])
        )
    print("-" * 80)

print("=" * 80)
print("Global Method Wins")
print(",".join(method_names))
for criterion in criteria:
    print(
        ",".join(
            [
                str(round(global_method_wins[criterion][method] / all_questions, 4))
                for method in method_names
            ]
        )
    )
print("-" * 80)
