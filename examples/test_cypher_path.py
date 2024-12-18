import logging
from typing import List
import boto3

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)


DATASET_DIR = "datasets/maple/Physics"
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

User questions are about node inquries which involve multi-hop relation paths.

---Goal---

Generate a trustful reasoning path that can be executed on graph using the given user question, based on the given schema of the knowledge graph.

Also give the cypher query that can be executed on the graph to get the answer. All entity labels are in the format "___nano_graphrag_bedrock_and_neo4j_Physics__chunk_entity_relation:entity_type".

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

Please use "-" when discribing the "paper" relation between paper node and author node (same for paper node and venue node) in the cypher query.
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
    # question = "What are the papers that cite the paper with title 'cosmology unique or not unique'?"
    # question = "What are the academic collaborators of the author who writes the paper 'cosmology unique or not unique'?"
    # question = "What are the papers of D. T. Woods that cite the papers in the venue where the paper with title 'cosmology unique or not unique' is published?"
    # question = "What are the papers that references the papers of D. T. Woods, constraint to D. T. Woods' papers that cite the papers in the venue where the paper with title 'cosmology unique or not unique' is published?"
    # question = "Who are the authors of the paper '{}'?"
    # question = "Where is the paper '{}' published?"
    # question = "What paper have the author '{}' published?"
    # question = "What are the academic collaborators of '{}'?"
    # question = "What venues have the author '{}' published in?"
    # question = "Who are the academic collaborators of the author who writes the paper '{}'?"
    # question = "What venues have the author of the paper '{}' published in?"
    question = "What venues have the academic collaborators of the author who writes the paper '{}' published in?"
    response = bedrock_generator(question, system)
    print_outputs(response)


if __name__ == "__main__":
    gen_paths()
