export CUDA_VISIBLE_DEVICES=0
export THRESHOLD=0.7 #Only useful when METHOD=="chunk_threshold", otherwise ignored.
METHOD="rag_baseline" #specify the RAG system
LLM="meta-llama/Llama-3.2-3B-Instruct" #Specify the Inference LLM
DATA="data/crag_task_1_dev_v4_release.jsonl.bz2" #Datapath
SPLIT=1 #Split
python generate.py \
      --dataset_path $DATA \
      --split $SPLIT \
      --model_name $METHOD \
      --llm_name $LLM