import json
import csv
from openai import OpenAI

PROMPT = """
**Prompt:**
You are an intelligent assistant tasked with classifying questions into different types based on the missing parts of a fact and the nature of the aspect being asked.

We have four types of questions, where s is the subject, p is the predicate, o is the object, and * means the missing part. We first define the four types of questions and then provide examples for each type.

The four types of questions are:
- **<s,*,*> (0):**
  In this type, the object and concrete predicate are both missing in the facts and question asks about the general or abstract information of the subject (e.g., themes, concepts, or a general description).
  Examples:
  - "Tell me about 'The Woman in Black: A Ghost Play'."
  - "Tell me about 'Frankenstein, or The Modern Prometheus'."
  - "Who is 'Barack Obama'?"
  - "Who is flo from progressive?"
- **<s,p,*> (1):**
  In this type, the object is missing in the facts and question focuses on finding the object, which is some specific, concrete details related to the subject (e.g., names, authors, publishers, or other direct attributes) with a concrete predicate p.
  Examples:
  - "Who are the authors of the book 'Sunshine for the Latter-Day Sa'?"
  - "What series have the author of the book 'Cookies for the Dragon (Saint Lakes, #2.1)' published?"
  - "What are the authors of the books that are published by the publishers that have published books of the series 'Shifter Justice'?"
- **<s,*,o> (2):**
  In this type, both the subject and object are given but the predicate is missing. Question asks about their relationships or interactions (e.g., conceptual connections, influences).
  Examples:
  - "What is the relationship between 'Kizuna' and 'Kazuma Kodaka'?"
  - "What is the relationship between 'Monster House Press' and 'Matt Hart'?"
  - "What are the impacts of information communication technology to public relation practices?"
  - "How are the directions of the velocity and force vectors related in a circular motion?"
  - "HOW AFRICAN AMERICANS WERE IMMIGRATED TO THE US?"
- **<s,p,o> (3):**
  In this question, the subject, predicates object are all given. Questions either inquires about particular aspects of the relationship between the entities or verifies whether a specific relationship exists.
  Examples:
  - "Have the authors 'Rubem Fonseca' and 'Lygia Fagundes Telles' ever published books in the same publishers? If so, tell me some examples."
  - "Do the publishers 'Scholastic Inc.' and 'Klutz' have any authors publishing books in both of them and what are the publications and authors?"

**Instruction:**
When given a question, analyze it based on the definitions above. If the question belongs to any of the four types, then return **only a single number** corresponding to the type of the question:

- **0** for <s,*,*>
- **1** for <s,p,*>
- **2** for <s,*,o>
- **3** for <s,p,o>

Note that the question may not belong to any of the types, in which case you should return **-1**.
For thoses cases, if they can be decomposed into multiple step where some step the sub-question belongs to one of the four types above and other steps are non-traversal operators like "comparing entity attributes" or "doing a sorting", then you should also provide a explanation for this on how it can be decomposed and return as follows:
- **-1: [Explanation of decomposition]**

For example:
- "What are the 5 biggest cities in the usa?": 1. the first step is a <s,p,*> question, traverse from the USA and find all the cities. 2. non-traversal operator: get the population of the city and do a sorting.
- "During what war did abraham lincoln serve as president?": 1. the first step is a <s,*,o> question, find the links between Lincoln and the president. 2. non-traversal operator: get the time slot (maybe on the edge attributes). 3. non-traversal operator: scan all wars and filter those in the time slot.
- "which city held the summer olympics twice?": 1. the first step is a <s,p,*> question, traverse from summer olympics to the cities. 2. non-traversal operator: count which cities appear twice in the results.

**You don't need to give explanation when the question falls into the above four types, only give explanation when it does not.**

**Question:** {}

**Your Answer:**
"""


client = OpenAI(
    api_key="sk-1326b87168a048e48cdc9356667f3acc", base_url="https://api.deepseek.com"
)

# Load the tsv file
file_path = "datasets/WikiQA.tsv"

questions = {}
# Open and read the TSV file
with open(file_path, "r") as file:
    reader = csv.DictReader(file, delimiter="\t")
    for row in reader:
        id = row["QuestionID"]
        if id not in questions:
            questions[id] = row["Question"]

# Access the data
print(f"Number of questions: {len(questions)}")

output_file = "results/WIKIQA_result.jsonl"

for q in questions.values():
    query = PROMPT.format(q)
    response = (
        client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": query}],
            stream=False,
        )
        .choices[0]
        .message.content
    )
    print(f"Question: {q}")
    print(f"Response: {response}")
    print("")
    # Write to file
    with open(output_file, "a") as file:
        file.write(json.dumps({"response": response, "question": q}) + "\n")
