export CUDA_VISIBLE_DEVICES=0
METHOD="rag_baseline" #specify the RAG system
LLM="meta-llama/Llama-3.2-3B-Instruct" #Specify the Inference LLM
DATA="data/crag_task_1_dev_v4_release.jsonl.bz2" #Datapath
RETRIES=10 #How many Retries we use
python evaluate.py \
      --dataset_path $DATA \
      --model_name $METHOD \
      --gen_llm_name $LLM \
      --max_retries $RETRIES
