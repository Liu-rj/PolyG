import jsonlines


# LLM_JUDGE_PATH = "../results/Physics/judgements.jsonl"
LLM_JUDGE_PATH = "../results/goodreads/judgements.jsonl"


judgements = []
with jsonlines.open(LLM_JUDGE_PATH, "r") as f:
    for item in f:
        judgements.append(item)

print(f"Number of judgements: {len(judgements)}")


question_types = [
    "single_entity_abstract",
    "single_entity_concrete",
    "multi_entity_abstract",
    "multi_entity_concrete",
]
question_judgement = {key: [] for key in question_types}
for judgement in judgements:
    question_judgement[judgement["question_type"]].append(judgement)

criteria = ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]
method_names = [
    "BFS",
    "Fastgraphrag_PPR",
    "shortest_paths",
    "cypher_single_entity",
    "cypher_multi_entity",
    "GraphCoT",
]
for question_type in question_types:
    method_wins = {name: {method: 0 for method in method_names} for name in criteria}
    for judgement in question_judgement[question_type]:
        for criterion in criteria:
            winner = judgement[criterion]["Winner"]
            for key in method_wins[criterion].keys():
                if key in winner:
                    method_wins[criterion][key] += 1

    for criterion, methods in method_wins.items():
        for method, value in methods.items():
            method_wins[criterion][method] = value / len(question_judgement[question_type])

    print("=" * 80)
    print(f"Question Type: {question_type}")
    print(method_wins)
    
    # print it as csv format
    print(",".join([""] + method_names))
    for criterion in criteria:
        print(",".join([str(method_wins[criterion][method]) for method in method_names]))
    print("-" * 80)
