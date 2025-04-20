# import jsonlines

# answers = []
# with open(
#     "/home/ubuntu/graphrag_planner/examples/results/goodreads/results_archive.jsonl",
#     "r",
# ) as f:
#     for item in jsonlines.Reader(f):
#         if item["method"] == "BFS":
#             continue
#         answers.append(item)

# with jsonlines.open(
#     "/home/ubuntu/graphrag_planner/examples/results/goodreads/results.jsonl", "w"
# ) as writer:
#     for row in answers:
#         writer.write(row)


import boto3

client = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

initial_message = [
    {"role": "user", "content": [{"text": "Who won the World Series in 2024?"}]}
]

response1 = client.converse(modelId=CHAT_MODEL_ID, messages=initial_message)

print("First response:")
print(response1["output"]["message"]["content"][0]["text"])


follow_up_message = [
    {
        "role": "user",
        "content": [{"text": "Can you tell me more about their performance?"}],
    }
]

response2 = client.converse(modelId=CHAT_MODEL_ID, messages=follow_up_message)

print("Second response without history:")
print(response2["output"]["message"]["content"][0]["text"])


# # Append the assistant's response from the first call
# assistant_response = response1["output"]["message"]
# conversation_history = initial_message + [assistant_response] + follow_up_message

# response3 = client.converse(
#     modelId=CHAT_MODEL_ID,
#     messages=conversation_history
# )

# print("Second response with history:")
# print(response3["output"]["message"]["content"][0]["text"])
