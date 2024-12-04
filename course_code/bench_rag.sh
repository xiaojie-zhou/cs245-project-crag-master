export CUDA_VISIBLE_DEVICES=3
python generate.py \
      --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
      --split 1 \
      --model_name "rag_baseline"