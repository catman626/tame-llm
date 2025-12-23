# rm -rf my 
# mkdir my
# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-0.5B --debug-mode output_hidden --gpu-batch-size 1 --num-gpu-batches 1 --gen-len 1

#### basic test
# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-0.5B  


# model=Qwen/Qwen2-0.5B
# model=Qwen/Qwen2-0.5B-Instruct 
# model=Qwen/Qwen2-7B
model="/home/llmserver/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B/snapshots/91d2aff3f957f99e4c74c962f2f408dcc88a18d8"

python -m flexllmgen.flex_qwen2 \
	--model $model \
	--gpu-batch-size 8 \
	--num-gpu-batches 2 \
	--prompt-len 2100 \
	--gen-len 256 \
	--sparse-mode block \
	--sep-layer False \
	--log-file log
	# --sparse-mode naive --attn-sparsity 0.1
	


# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-0.5B-Instruct --prompt-len 2048 --gen-len 256

# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-0.5B --attn-sparsity 0.1 --input-file prompt01.txt --gen-len 128
# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-7B 
# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-0.5B  --input-file prompt01.txt --gen-len 128
# python -m flexllmgen.flex_opt --model facebook/opt-125m --attn-sparsity 0.1
