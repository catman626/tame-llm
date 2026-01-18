model="/home/llmserver/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B/snapshots/91d2aff3f957f99e4c74c962f2f408dcc88a18d8"
model="/home/llmserver/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

gpu_bs=1
gen_len=2
prompt_len=100000

python -m flexllmgen.flex_qwen2 \
	--model $model \
	--gpu-batch-size $gpu_bs \
	--num-gpu-batches 2 \
	--log-file log \
	--prompt-len $prompt_len \
	--gen-len $gen_len \
	--sparse-mode block