import os
import argparse
import jsonlines
import random
import boto3
import re
from typing import List
from openai import OpenAI
from pydantic import SecretStr
from dotenv import load_dotenv

load_dotenv("../.env")

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--benchmark_dir", type=str, default="benchmarks/physics", required=True
)
args = argparser.parse_args()

client = OpenAI(
    api_key=SecretStr(os.getenv("DEEPSEEK_API_KEY")),  # type: ignore
    base_url="https://api.deepseek.com",
)

PROMPT = """
You are a linguistics expert, please paraphrase the given question without changing any semantic meaning of it.
For example, "Give a broad discription of the the finder of 'Higgs boson'." is not equal to "Who is the finder of 'Higgs boson'?", since the former one asks about general information while what the latter inquires about is vague and can just be the finder name.
Also, "ave Issac Newton and Albert Einstein both contributed to the same same field of science?" is not equal to "What is the field of science that both Issac Newton and Albert Einstein contributed to?", since the former one is checking if the relation exists while the latter inquires about concrete entities and can just be the field name.

Note that the contents in '' (single quotes) is an exact entity name which will be used to match entities in the knowledge graph, so do not change any word in the single quote '' part.
Also, in your returned paraphrased question, please keep the contents in '' part unchanged.

Question: {}:

You are required to return 4 paraphrased questions in the following format (enclose the question in square brackets):
1. [<paraphrased question1>]
2. [<paraphrased question2>]
3. [<paraphrased question3>]
4. [<paraphrased question4>]
"""

ERROR_PROMPT = """
When generating the paraphrased questions, you have made a mistake and caused an error.

Your previous response: {response}

Error: {error}

Please fix the error and generate the paraphrased questions again.

You are required to return 4 paraphrased questions in the following format (enclose the question in square brackets):
1. [<paraphrased question1>]
2. [<paraphrased question2>]
3. [<paraphrased question3>]
4. [<paraphrased question4>]
"""


print(f"Benchmark dir: {args.benchmark_dir}")

bedrock_cli = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"


def print_outputs(outputs):
    print("=" * 80)
    print("Generated reponse:")
    print(outputs)
    print("-" * 80)


def bedrock_generator(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: List[dict] = [],
) -> str:
    messages, system = [], []
    if system_prompt:
        system.append({"text": system_prompt})

    messages.extend(history_messages)
    messages.append({"role": "user", "content": [{"text": prompt}]})

    response = bedrock_cli.converse(modelId=MODEL_ID, messages=messages, system=system)
    return response["output"]["message"]["content"][0]["text"]


def openai_generator(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: List[dict] = [],
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="deepseek-chat", messages=messages, stream=False, temperature=1.0
    )
    return response.choices[0].message.content  # type: ignore


if __name__ == "__main__":
    question_types = [
        "subject_centered_raw",
        "object_discovery_raw",
        "predicate_discovery_raw",
        "fact_check_raw",
        "nested_question_raw",
    ]
    for question_type in question_types:
        output_file = os.path.join(args.benchmark_dir, f"{question_type}.jsonl")

        contents = []
        with open(os.path.join(args.benchmark_dir, f"{question_type}.jsonl"), "r") as f:
            for item in jsonlines.Reader(f):
                contents.append(item)

        for item in contents:
            question, id_mapping = item["question"], item["entity"]
            error_msg = ""

            key_strs = [f"'{key}'" for key in id_mapping.keys()]

            response = ""
            while True:
                try:
                    response = bedrock_generator(
                        PROMPT.format(question) + "\n\n" + error_msg
                    )
                    print(response)

                    paraphrased_questions = [question]
                    for line in response.split("\n"):
                        try:
                            match = re.match(r"^\d\.\s?\[", line)
                            if not match:
                                continue
                            new_question = line[match.end() :].strip().strip("]")
                        except IndexError as e:
                            print(f"Error parsing line: {line}, Error: {str(e)}")
                            continue

                        for key_str in key_strs:
                            if key_str not in new_question:
                                raise ValueError(
                                    f"Entity name {key_str} not found in rephrased question."
                                )
                        paraphrased_questions.append(new_question)

                    break
                except Exception as e:
                    print(f"Response: {response}, Error: {str(e)}")
                    error_msg = ERROR_PROMPT.format(response=response, error=str(e))

            print(paraphrased_questions)

            num = random.randint(0, 4)
            len_q = len(paraphrased_questions)
            if len_q < 5:
                print(f"Warning: Only {len_q} paraphrased questions generated")
                continue
            paraphrased_question = paraphrased_questions[num % len_q]
            print(f"Selected paraphrased question: {paraphrased_question}")

            rephrased_item = item.copy()
            rephrased_item["question"] = paraphrased_question

            # write the rephrased question to the output file
            with jsonlines.open(output_file, "a") as writer:
                writer.write(rephrased_item)
