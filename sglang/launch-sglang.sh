SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
--model-path Qwen/Qwen2.5-1.5B-Instruct \
--json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'  \
--context-length 131072


# SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
	# --model-path Qwen/Qwen2.5-7B-Instruct \
	# --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' \
	# --context-length 131072 \
	# --tensor-parallel-size 2
