# PolyG: Adaptive Graph Traversal for Diverse GraphRAG Questions

<div align="center">
  <img src="fig/hf-logo.svg" width="15" height="15" /> <a href="https://huggingface.co/datasets/Liu-rj/PolyBench">Dataset</a>&nbsp;&nbsp; | &nbsp;&nbsp;📖 <a href="https://arxiv.org/pdf/2504.02112">Arxiv</a>
</div>
&nbsp;

This repo provides the PolyBench and PolyG implementation of the paper [PolyG: Adaptive Graph Traversal for Diverse GraphRAG Questions](https://arxiv.org/abs/2504.02112).

<div align="center">
  <img src="fig/workflow.png" />
</div>

## Installation

* Create a new env with python 3.12:

```shell
conda create -n polyg python=3.12
```

* Install all dependencies:

```shell
pip install -r requirements.txt
```

* Install Neo4j 2025.03.0:

Refer to [Neo4j Installation](https://neo4j.com/docs/operations-manual/current/installation/).

* Install PolyG (from the root directory of this repo):

```shell
pip install -e .
```

## Dataset and Benchmark

* Install the dataset:

For the dataset (knowledge graphs), please refer to [RGBench](https://github.com/PeterGriffinJin/Graph-CoT).

Store the knowledge graphs into `datasets` directory (from the root directory of this repo).

* Convert the graphs into desired formats:

Go to `preprocess_dataset` directory, run

```shell
python preprocess_graph.py --path dataset/physics
python preprocess_graph.py --path dataset/goodreads
python preprocess_graph.py --path dataset/amazon
```

* Import the data to Neo4j:

At the `preprocess_dataset` directory, run

```shell
bash neo4j_bulk_insert.sh
```

* PolyBench:

Our proposed PolyBench is available in `benchmarks` directory and on [huggingface](https://huggingface.co/datasets/Liu-rj/PolyBench).

`[dataset_name].jsonl`, for example `physics.jsonl`, contains the full question set for each dataset.

We also provide seperate question set for each type (`*_raw.jsonl` contains the unparaphrased version):

* `subject_centered.jsonl` contains the question set for type `<s,*,*>`.

* `object_discovery.jsonl` contains the question set for type `<s,p,*>`.

* `predicate_discovery.jsonl` contains the question set for type `<s,*,o>`.

* `fact_check.jsonl` contains the question set for type `<s,p,o>`.

* `nested_question.jsonl` contains the question set for type `nested`.

## Use PolyG

* Run a toy example (on the physics graph):

At the `examples` directory, run

```shell
python example.py --model openai/gpt-4o --data_dir ../datasets/physics
```

* Run end-to-end evaluation on PolyBench:

At the `examples` directory, run

```shell
python experiment.py --model openai/gpt-4o --data_dir ../datasets/physics --benchmark_dir ../benchmarks/physics
python experiment.py --model openai/gpt-4o --data_dir ../datasets/goodreads --benchmark_dir ../benchmarks/goodreads
python experiment.py --model openai/gpt-4o --data_dir ../datasets/amazon --benchmark_dir ../benchmarks/amazon
```

Results will be stored in `examples/results/[graph name]/[model name]/results.jsonl`.

To evaluate the results, checkout to `examples/evaluation` directory, run:

```shell
# for win rates
python judge_by_llm.py --model openai/gpt-4o --dataset physics
python judge_by_llm.py --model openai/gpt-4o --dataset goodreads
python judge_by_llm.py --model openai/gpt-4o --dataset amazon

# for F1-score, Precision, Recall, Accuracy and Hit
python compute_f1_hit.py --model openai/gpt-4o --dataset physics
python compute_f1_hit.py --model openai/gpt-4o --dataset goodreads
python compute_f1_hit.py --model openai/gpt-4o --dataset amazon
```

Results will be saved in `examples/results/[graph name]/[model name]/judegments.jsonl` and `examples/results/[graph name]/[model name]/detailed_evaluation.jsonl`.
