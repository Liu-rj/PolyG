import csv
import os
from collections import defaultdict

PREFIX = "../question_classifier/"
RESULTS_PATH = ["physics_results.csv", "goodreads_results.csv", "amazon_results.csv"]

# read csv file
dataset_results = {"physics": [], "goodreads": [], "amazon": []}

for path in RESULTS_PATH:
    with open(os.path.join(PREFIX, path), newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset_results[path.split("_")[0]].append(row)

for dataset, results in dataset_results.items():
    print(f"Dataset: {dataset}")
    total_time = defaultdict(float)
    total_tokens = defaultdict(float)
    total_counts = defaultdict(int)
    correct = 0
    for result in results:
        if result["prediction"] == result["type"]:
            correct += 1
        total_time[result["prediction"]] += float(result["latency"])
        total_tokens[result["prediction"]] += float(result["num_tokens"])
        total_counts[result["prediction"]] += 1
    print(f"Accuracy: {correct}/{len(results)}")
    for key, value in total_time.items():
        print(f"Average time for {key}: {value / total_counts[key]}")
    for key, value in total_tokens.items():
        print(f"Average tokens for {key}: {value / total_counts[key]}")

    for key, value in total_time.items():
        print(value / total_counts[key])
    for key, value in total_tokens.items():
        print(value / total_counts[key])

    print()
