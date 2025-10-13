model=claude-3.5-sonnet
model=deepseek-chat
model=Qwen/Qwen3-14B

python experiment.py --benchmark=webqsp --model=$model

python experiment.py --benchmark=cwq --model=$model
