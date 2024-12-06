export CUDA_VISIBLE_DEVICES=0
python evaluate.py \
    --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
    --model_name "chunk_threshold" \
    --llm_name "meta-llama/Llama-3.2-3B-Instruct"