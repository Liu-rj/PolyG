import logging
from typing import List
import boto3

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)


DATASET_DIR = "datasets/maple/physics"
WORKING_DIR = f"./nano_graphrag_vllm_and_local_embedding_{DATASET_DIR.split('/')[-1]}"
MAX_MODEL_LEN = 128000
MAX_CONTEXT_TOKENS = 100000
MAX_OUTPUT_TOKENS = 5000

print("Dataset dir:", DATASET_DIR, "Working dir:", WORKING_DIR)

CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

system = """---Role---

You are a helpful assistant that can generate trustful reasoning paths with the knowledge of the graph schema to answer the given user question.

---Setting---

You will be provided with the knowledge graph schema, which indicates the types of nodes and edges, and by what relations nodes are connected.

User questions are about relationships between nodes which can be indirect and result in multi-hop relation paths. User questions may include a vague relation constraints that indicates some specific paths.

An example:
User question is "What is the relationship between 'J. Koll' and 'Z. Staykova' in terms of paper reference?". In this case, the user is asking for the paths between two authors that is constraint to paper references and citations.
And the cypher query for this question is (where only paths that contains reference and citation relations should be considered):
```cypher
MATCH (author1 {name: 'J. Koll'}), (author2 {name: 'Z. Staykova'})
MATCH path = (author1)-[:paper]-(paper1)-[:reference|cited_by]-(paper2)-[:paper]-(author2)
RETURN path
ORDER BY length(path)
LIMIT 10
```

---Goal---

Generate a trustful reasoning path that can be executed on graph using the given user question, based on the given schema of the knowledge graph. Also give the cypher query that can be executed on the graph to get the answer.

If you don't have adequate information to give trustful reasoning paths, just say so. Do not make anything up.

Do not include information where the supporting evidence for it is not provided.

---Graph Schema---

Definition of the graph:
This knowledge graph is a citation graph in physics, there are three types of nodes in this graph: paper, author and venue.

Node properties:
1. type: author, properties: ["name", "id", "node_type"]
2. type: paper, properties: ["name", "label", "year", "id", "node_type", "abstract"]
3. type: venue, properties: ["name", "id", "node_type"]

Edge properties:
Author nodes are linked to their paper nodes by authorship. Specific relations are:
1. author_node -> "paper" -> paper_node

Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Specific relations are:
1. paper_node -> "paper" -> author_node
2. paper_node -> "reference" -> paper_node
3. paper_node -> "paper" -> venue_node
3. paper_node -> "cited_by" -> venue_node

Venue nodes are linked to their included paper nodes. Specific relations are:
1. venue_node -> "paper" -> paper_node

Please use "-" when discribing the "paper" relation between paper node and author node (same for paper node and venue node) in the cypher query. Return the complete paths if possible.
"""


def print_outputs(outputs):
    print("=" * 80)
    print("Generated reponse:")
    print(outputs)
    print("-" * 80)


def bedrock_generator(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[dict] = [],
    **kwargs,
) -> str:
    bedrock_cli = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
    )

    messages, system = [], []
    if system_prompt:
        system.append({"text": system_prompt})

    messages.append({"role": "user", "content": [{"text": prompt}]})

    response = bedrock_cli.converse(
        modelId=CHAT_MODEL_ID, messages=messages, system=system
    )
    return response["output"]["message"]["content"][0]["text"]


def gen_paths():
    # question = "What is the relationship between 'J. Koll' and 'Z. Staykova' in terms of paper reference?"
    # question = "What is the relationship between 'J. Koll' and 'Z. Staykova' in terms of common collaborators?"
    # question = "What is the relationship between 'J. Koll' and 'Z. Staykova' in terms of venues and papers in those venues?"

    question = "What is the relationshop between the authors '{}' and '{}' regarding collaborated papers?"
    response = bedrock_generator(question, system)
    print_outputs(response)


if __name__ == "__main__":
    gen_paths()
