python experiment.py --benchmark=webqsp --model=Qwen/Qwen3-14B
python experiment.py --benchmark=webqsp --model=deepseek/deepseek-chat

python experiment.py --benchmark=cwq --model=Qwen/Qwen3-14B
python experiment.py --benchmark=cwq --model=deepseek/deepseek-chat

python compute_f1_hit.py --dataset=webqsp --model=Qwen/Qwen3-14B
python compute_f1_hit_graphcot.py --dataset=webqsp --model=Qwen/Qwen3-14B
python compute_f1_hit.py --dataset=webqsp --model=deepseek/deepseek-chat
python compute_f1_hit_graphcot.py --dataset=webqsp --model=deepseek-chat

python compute_f1_hit.py --dataset=cwq --model=Qwen/Qwen3-14B
python compute_f1_hit_graphcot.py --dataset=cwq --model=Qwen/Qwen3-14B
python compute_f1_hit.py --dataset=cwq --model=deepseek/deepseek-chat
python compute_f1_hit_graphcot.py --dataset=cwq --model=deepseek-chat
