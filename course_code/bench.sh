export CUDA_VISIBLE_DEVICES=0
python generate.py \
      --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
      --split 1 \
      --model_name "file_levl_rag" \
      --llm_name "meta-llama/Llama-3.2-1B-Instruct"