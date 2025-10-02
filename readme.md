# PolyG: Adaptive Graph Traversal for Diverse GraphRAG Questions

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

Go to `examples/preprocess` directory, run

```shell
python preprocess_graph.py --path dataset/physics
python preprocess_graph.py --path dataset/goodreads
python preprocess_graph.py --path dataset/amazon
```

* Import the data to Neo4j:

At the `examples/preprocess` directory, run

```shell
bash neo4j_bulk_insert.sh
```

* PolyBench:

Our proposed PolyBench is available at `benchmarks` directory.

## Run the experiments

The scripts to reproduce the experimental results are provide in `examples\run.sh`.
