export CUDA_VISIBLE_DEVICES=1
python generate.py \
      --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
      --split 1 \
      --model_name "vanilla_baseline" \
      --llm "meta-llama/Llama-3.1-8B-Instruct"