import jsonlines


# LLM_JUDGE_PATH = "../results/Physics/claude-3.5-sonnet/judgements_rephrased_new.jsonl"
# LLM_JUDGE_PATH = "../results/goodreads/claude-3.5-sonnet/judgements_rephrased_new.jsonl"
LLM_JUDGE_PATH = "../results/amazon/claude-3.5-sonnet/judgements_rephrased_new.jsonl"

# LLM_JUDGE_PATH = "../results/Physics/claude-3.5-sonnet/judgements_rephrased.jsonl"
# LLM_JUDGE_PATH = "../results/goodreads/gpt-4o-mini/judgements_rephrased.jsonl"
# LLM_JUDGE_PATH = "../results/amazon/claude-3.5-sonnet/judgements_rephrased.jsonl"


judgements = []
with jsonlines.open(LLM_JUDGE_PATH, "r") as f:
    for item in f:
        judgements.append(item)

print(f"Number of judgements: {len(judgements)}")


question_types = [
    "single_entity_abstract_rephrased",
    "single_entity_concrete_rephrased",
    "multi_entity_abstract_rephrased",
    "multi_entity_concrete_rephrased",
    "nested_question_rephrased",
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
    # "shortest_paths",
    # "cypher_multi_entity",
    "cypher_only",
    "adaptive",
]
global_method_wins = {name: {method: 0 for method in method_names} for name in criteria}
all_questions = 0
for question_type in question_types:
    method_wins = {name: {method: 0 for method in method_names} for name in criteria}
    for judgement in question_judgement[question_type]:
        for criterion in criteria:
            winner = judgement[criterion]["Winner"]
            for key in method_wins[criterion].keys():
                if key in winner:
                    method_wins[criterion][key] += 1
                    global_method_wins[criterion][key] += 1

    all_questions += len(question_judgement[question_type])
    for criterion, methods in method_wins.items():
        for method, value in methods.items():
            if len(question_judgement[question_type]) > 0:
                method_wins[criterion][method] = value / len(
                    question_judgement[question_type]
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
        for method in method_names:
            global_method_wins[criterion][method] += method_wins[criterion][method]
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
