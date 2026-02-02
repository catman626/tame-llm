vllm serve Qwen/Qwen2.5-72B-Instruct \
    --pipeline-parallel-size 4  \
    --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072
