python experiment.py --data_dir=datasets/maple/Physics --benchmark_dir=benchmarks/physics --model=claude-3.5-sonnet
python evaluation/judege_by_llm.py --dataset=physics
python evaluation/compute_f1.py --dataset=physics

python experiment.py --data_dir=datasets/goodreads --benchmark_dir=benchmarks/goodreads --model=claude-3.5-sonnet
python evaluation/judege_by_llm.py --dataset=goodreads
python evaluation/compute_f1.py --dataset=goodreads

python experiment.py --data_dir=datasets/amazon --benchmark_dir=benchmarks/amazon --model=claude-3.5-sonnet
python evaluation/judege_by_llm.py --dataset=amazon
python evaluation/compute_f1.py --dataset=amazon
