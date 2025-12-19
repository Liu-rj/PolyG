import jsonlines
import os

dataset = "amazon"
question_types = {
    "single_entity_abstract": "<s,*,*>",
    "single_entity_concrete": "<s,p,*>",
    "multi_entity_abstract": "<s,*,o>",
    "multi_entity_concrete": "<s,p,o>",
    "nested_question": "nested",
}
id = 0
contents = []
for question_type, abbr in question_types.items():
    with open(os.path.join(dataset, f"{question_type}.jsonl"), "r") as f:
        for item in jsonlines.Reader(f):
            item["qid"] = id
            id += 1
            item["type"] = abbr
            if item["answer"] == "N/A":
                item["answer"] = []

            entities = []
            for name, nid in item["entity"].items():
                entities.append((name, nid))
            item["entity"] = entities
            contents.append(item)

with open(os.path.join(dataset, f"{dataset}.jsonl"), "w") as f:
    writer = jsonlines.Writer(f)
    writer.write_all(contents)
    writer.close()
