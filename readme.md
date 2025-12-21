# PolyG: Adaptive Graph Traversal for Diverse GraphRAG Questions

<div align="center">
    <a href="https://huggingface.co/datasets/Liu-rj/PolyBench" target="_blank"><img src="https://img.shields.io/badge/Benchmark-Hugging Face-orange?logo=huggingface"></a>
    <a href="https://arxiv.org/pdf/2504.02112" target="_blank"><img src="https://img.shields.io/badge/Paper-Arxiv-red?logo=arxiv"></a>
    <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/last-commit/Liu-rj/PolyG?color=blue"/></a>
</div>

This repo provides the PolyBench and PolyG implementation of the paper [PolyG: Adaptive Graph Traversal for Diverse GraphRAG Questions](https://arxiv.org/abs/2504.02112).

<div align="center">
  <img src="fig/workflow.png" />
</div>

GraphRAG enhances LLMs to generate quality answers for user questions by retrieving related facts from external knowledge graphs. However, current GraphRAG methods are primarily evaluated on and overly tailored for KGQA benchmarks, which are biased towards a few specific question patterns and do not reflect the diversity of realworld questions.

To better evaluate GraphRAG methods, we propose a complete four-class taxonomy to categorize the basic patterns of knowledge graph questions and use it to create PolyBench, a new GraphRAG benchmark encompassing a comprehensive set of graph questions.

With the new benchmark, we find that different question patterns require distinct graph traversal strategies and context formation. This motivates us to propose PolyG, an adaptive GraphRAG approach by decomposing and categorizing the questions according to our proposed question taxonomy. Built on top of a unified interface and execution engine, PolyG dynamically prompts the LLM to generate a graph database query to retrieve the context for each decomposed basic question.

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

## Use PolyBench

PolyBench is created on three real-world knowledge graphs from [GRBench](https://huggingface.co/datasets/PeterJinGo/GRBench) across the academia, literature and e-commerce domain.
With a total of 1,200 questions instantiated from 73 well-crafted question templates, PolyBench consists of 400 questions for each graph spanning 5 question patterns.
This comprehensive set of question patterns is derived from a complete four-class taxonomy under the triplet format of the knowledge graph, including four basic patterns and nested types:

- `<s,*,*>` (subject centered): Questions about an entity (the subject) with no specific relation constraints (the predicate) and target entity (the object). The task is to answer a general question about the entity. Example: “Who is Isaac Newton?”
- `<s,p,*>` (object discovery): Questions about an entity (the subject) with specific relation constraints (the predicate) but misses the target entity (the object). The task is to answer one specific aspect of the entity. Example: “What theories and principles has Isaac Newton developed?”
- `<s,*,o>` (predicate discovery): Questions about any relations (the predicate) between two entities (the subject and object). The task is to provide the relations between two entities. Example: “How is Isaac Newton and Albert Einstein related?”
- `<s,p,o>` (fact check): Questions about specific relations (the predicate) between two entities (the subject and object). The task is to check the existence of a specific relationship between the two entities. Example: “Have Isaac Newton and Albert Einstein both contributed to the same same field of science?”
- nested types: Complex questions involving the nesting of multiple basic questions. For example, nested questions can be about general information of an unknown entity with specific relation constraints. Example: “Tell me about the scientist who developed univsersal gravitation.”


### Process the dataset

* Download the raw graphs

For the dataset (knowledge graphs), please refer to [GRBench](https://github.com/PeterGriffinJin/Graph-CoT). We select the `physics` graph from the academia domain, `goodreads` graph from the literature domain, and `amazon` from the e-commerce domain.

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

### Use the benchmark

Our proposed PolyBench is available in `benchmarks` directory and on [huggingface](https://huggingface.co/datasets/Liu-rj/PolyBench).

`[dataset_name].jsonl`, for example `physics.jsonl`, contains the full question set for each dataset.

We also provide seperate question set for each type (`*_raw.jsonl` contains the unparaphrased version):

* `subject_centered.jsonl` contains the question set for type `<s,*,*>`.

* `object_discovery.jsonl` contains the question set for type `<s,p,*>`.

* `predicate_discovery.jsonl` contains the question set for type `<s,*,o>`.

* `fact_check.jsonl` contains the question set for type `<s,p,o>`.

* `nested_question.jsonl` contains the question set for type `nested`.

Or from [huggingface](https://huggingface.co/datasets/Liu-rj/PolyBench), use the question set from each domain:

```
from datasets import load_dataset
domain = "physics" # can be selected from [physics (academia), goodreads (literature), amazon (e-commerce)]
dataset = load_dataset("Liu-rj/PolyBench", domain, split="test")
```

## Use PolyG

PolyG is a general and effective GraphRAG solution designed to handle a wide range of graph questions. PolyG automatically classifies and decomposes user questions into a sequence of basic sub-questions. For each sub-question, it generates an appropriate Cypher query with adaptive prompting and self-correction mechanisms to retrieve relevant information. The responses to these sub-questions are then aggregated and fed into the LLM to produce a final answer.

PolyG comprises the following five execution stages: _question categorization_, _question decomposition_, _Cypher query generation_, _query execution_ and _context formation_.

### Usage

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

## LICENSE

```
Copyright 2025 Renjie Liu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Cite PolyG

If you find PolyG and PolyBench helpful, please cite the paper.

```
@misc{liu2025polygadaptivegraphtraversal,
      title={PolyG: Adaptive Graph Traversal for Diverse GraphRAG Questions}, 
      author={Renjie Liu and Haitian Jiang and Xiao Yan and Bo Tang and Jinyang Li},
      year={2025},
      eprint={2504.02112},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.02112}, 
}
```
