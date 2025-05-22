# cd /home/ubuntu/PolyG/examples

# python experiment.py --data_dir=datasets/maple/Physics --benchmark_dir=benchmarks/physics --model=claude-3.5-sonnet

# cd /home/ubuntu/fast-graphrag/examples

# python bedrock.py --data_dir=/home/ubuntu/PolyG/examples/datasets/maple/Physics --benchmark_dir=/home/ubuntu/PolyG/examples/benchmarks/physics

# cd /home/ubuntu/Graph-CoT/Graph-CoT/

# OPENAI_KEY=xxx
# GPT_version=claude-3-5-sonnet
# max_steps=20

# DATASET=maple
# SUBDATASET=Physics # Biology, Chemistry, Materials_Science, Medicine, Physics
# DATA_PATH=/home/ubuntu/PolyG/examples/datasets/maple/$SUBDATASET
# BENCHMARK=/home/ubuntu/PolyG/examples/benchmarks/physics
# SAVE_FILE=/home/ubuntu/Graph-CoT/Graph-CoT/results/$GPT_version/maple-$SUBDATASET/results.jsonl

# CUDA_VISIBLE_DEVICES=0 python code/run.py --dataset $DATASET \
#                     --path $DATA_PATH \
#                     --benchmark_dir $BENCHMARK \
#                     --save_file $SAVE_FILE \
#                     --llm_version $GPT_version \
#                     --openai_api_key $OPENAI_KEY \
#                     --max_steps $max_steps

# cd /home/ubuntu/PolyG/examples/evaluation

# python judge_by_llm.py --dataset=physics




# cd /home/ubuntu/PolyG/examples

# python experiment.py --data_dir=datasets/goodreads --benchmark_dir=benchmarks/goodreads --model=claude-3.5-sonnet

# cd /home/ubuntu/fast-graphrag/examples

# python bedrock.py --data_dir=/home/ubuntu/PolyG/examples/datasets/goodreads --benchmark_dir=/home/ubuntu/PolyG/examples/benchmarks/goodreads

# cd /home/ubuntu/Graph-CoT/Graph-CoT/

# OPENAI_KEY=xxx
# GPT_version=claude-3-5-sonnet
# max_steps=20

# DATASET=goodreads # legal, biomedical, amazon, goodreads, dblp
# DATA_PATH=/home/ubuntu/PolyG/examples/datasets/$DATASET
# BENCHMARK=/home/ubuntu/PolyG/examples/benchmarks/$DATASET
# SAVE_FILE=/home/ubuntu/Graph-CoT/Graph-CoT/results/$GPT_version/$DATASET/results.jsonl

# CUDA_VISIBLE_DEVICES=0 python code/run.py --dataset $DATASET \
#                     --path $DATA_PATH \
#                     --benchmark_dir $BENCHMARK \
#                     --save_file $SAVE_FILE \
#                     --llm_version $GPT_version \
#                     --openai_api_key $OPENAI_KEY \
#                     --max_steps $max_steps

# cd /home/ubuntu/PolyG/examples/evaluation

# python judge_by_llm.py --dataset=goodreads




# cd /home/ubuntu/PolyG/examples

# python experiment.py --data_dir=datasets/amazon --benchmark_dir=benchmarks/amazon --model=claude-3.5-sonnet

# cd /home/ubuntu/fast-graphrag/examples

# python bedrock.py --data_dir=/home/ubuntu/PolyG/examples/datasets/amazon --benchmark_dir=/home/ubuntu/PolyG/examples/benchmarks/amazon

# cd /home/ubuntu/Graph-CoT/Graph-CoT/

# OPENAI_KEY=xxx
# GPT_version=claude-3-5-sonnet
# max_steps=20

# DATASET=amazon # legal, biomedical, amazon, goodreads, dblp
# DATA_PATH=/home/ubuntu/PolyG/examples/datasets/$DATASET
# BENCHMARK=/home/ubuntu/PolyG/examples/benchmarks/$DATASET
# SAVE_FILE=/home/ubuntu/Graph-CoT/Graph-CoT/results/$GPT_version/$DATASET/results.jsonl

# CUDA_VISIBLE_DEVICES=0 python code/run.py --dataset $DATASET \
#                     --path $DATA_PATH \
#                     --benchmark_dir $BENCHMARK \
#                     --save_file $SAVE_FILE \
#                     --llm_version $GPT_version \
#                     --openai_api_key $OPENAI_KEY \
#                     --max_steps $max_steps

# cd /home/ubuntu/PolyG/examples/evaluation

# python judge_by_llm.py --dataset=amazon




python experiment.py --data_dir=datasets/maple/Physics --benchmark_dir=benchmarks/physics --model=claude-3.5-sonnet
python evaluation/judege_by_llm.py --dataset=physics
python evaluation/compute_f1.py --dataset=physics

python experiment.py --data_dir=datasets/goodreads --benchmark_dir=benchmarks/goodreads --model=claude-3.5-sonnet
python evaluation/judege_by_llm.py --dataset=goodreads
python evaluation/compute_f1.py --dataset=goodreads

python experiment.py --data_dir=datasets/amazon --benchmark_dir=benchmarks/amazon --model=claude-3.5-sonnet
python evaluation/judege_by_llm.py --dataset=amazon
python evaluation/compute_f1.py --dataset=amazon


python experiment.py --data_dir=datasets/maple/Physics --benchmark_dir=benchmarks/physics --model=gemini-2.0-flash
