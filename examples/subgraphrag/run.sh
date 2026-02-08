python experiment.py --benchmark=webqsp --model=Qwen/Qwen3-14B --path webqsp_Oct20-10:11:32/cpt.pth
python experiment.py --benchmark=webqsp --model=deepseek/deepseek-chat --path webqsp_Oct20-10:11:32/cpt.pth

python experiment.py --benchmark=cwq --model=Qwen/Qwen3-14B --path cwq_Dec20-02:44:51/cpt.pth
python experiment.py --benchmark=cwq --model=deepseek/deepseek-chat --path cwq_Dec20-02:44:51/cpt.pth

python experiment_adaptive.py --benchmark=webqsp --model=Qwen/Qwen3-14B --path webqsp_Oct20-10:11:32/cpt.pth
python experiment_adaptive.py --benchmark=webqsp --model=deepseek/deepseek-chat --path webqsp_Oct20-10:11:32/cpt.pth

python experiment_adaptive.py --benchmark=cwq --model=Qwen/Qwen3-14B --path cwq_Dec20-02:44:51/cpt.pth
python experiment_adaptive.py --benchmark=cwq --model=deepseek/deepseek-chat --path cwq_Dec20-02:44:51/cpt.pth
