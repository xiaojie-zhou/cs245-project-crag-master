export CUDA_VISIBLE_DEVICES=2
MODEL="file_levl_rag"
LLM="meta-llama/Llama-3.2-1B-Instruct"
python evaluate.py \
      --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
      --model_name $MODEL \
      --llm_name $LLM \
      --max_retries 10