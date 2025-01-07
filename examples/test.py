# from neo4j import GraphDatabase

# driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "123456789"))

# # Access driver configuration
# print(driver._config.max_connection_pool_size)  # Default: 100
# print(driver._config.connection_timeout)       # Default: 30 seconds

# driver.close()

# import csv
# import os
# import sys
# import json
# import jsonlines

# # Increase the CSV field size limit
# csv.field_size_limit(sys.maxsize)

# # read lines from results/Physics/results.csv
# with open("results/Physics/results.csv", "r") as f:
#     reader = csv.reader(f)
#     rows = list(reader)

# # load question answer pairs from datasets/maple/Physics
# question_types = [
#         "single_entity_abstract",
#         "single_entity_concrete",
#         "multi_entity_abstract",
#         "multi_entity_concrete",
#     ]
# contents = {}
# for question_type in question_types:
#     with open(os.path.join("datasets/maple/Physics", f"{question_type}.jsonl"), "r") as f:
#         for item in jsonlines.Reader(f):
#             contents[item["question"]] = item["answer"]

# # store toresults/Physics/results.jsonl in json formats
# with open("results/Physics/results.jsonl", "w") as f:
#     for row in rows:
#         result_entree = {
#             "question_type": row[0],
#             "question": row[1],
#             "method": row[2],
#             "model_answer": row[3],
#             "duration": row[4],
#             "token_count": row[5],
#             "api_calls": row[6],
#             "gt_answer": contents[row[1]],
#         }
#         f.write(json.dumps(result_entree) + "\n")


import jsonlines

new = {
    "question_type": "single_entity_concrete",
    "question": "Who are the academic collaborators of the author who writes the paper 'a simplified approach to collision processes'?",
    "method": "cypher_single_entity",
    "model_answer": "Dennis Sivers, David G. Richards, Jian-Wei Qui, Lionel E. Gordon, Mehrdad Goshtasbpour, David Richards, Jianwei Qiu, Asim Gangopadhyaya",
    "duration": 9.51,
    "token_count": 622,
    "api_calls": 1,
    "gt_answer": "Dennis Sivers, David G. Richards, Jian-Wei Qui, Lionel E. Gordon, Mehrdad Goshtasbpour, David Richards, Jianwei Qiu, Asim Gangopadhyaya",
}

# load the jsonl file and replace one line
lines = []
with jsonlines.open("results/Physics/results.jsonl", "r") as reader:
    for item in reader:
        if item["question"] == new["question"]:
            lines.append(new)
        else:
            lines.append(item)

# write the new jsonl file
with jsonlines.open("results/Physics/results.jsonl", "w") as writer:
    writer.write_all(lines)
