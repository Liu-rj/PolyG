python experiment.py --benchmark=webqsp --model=Qwen/Qwen3-14B --path webqsp_Oct20-10:11:32/cpt.pth
python experiment.py --benchmark=webqsp --model=deepseek/deepseek-chat --path webqsp_Oct20-10:11:32/cpt.pth

python experiment.py --benchmark=cwq --model=Qwen/Qwen3-14B --path /home/renjie/SubgraphRAG/retrieve/cwq_Dec20-02:44:51/cpt.pth --topk 200
python experiment.py --benchmark=cwq --model=deepseek/deepseek-chat --path /home/renjie/SubgraphRAG/retrieve/cwq_Dec20-02:44:51/cpt.pth
