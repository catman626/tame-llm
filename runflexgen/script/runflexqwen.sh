# rm -rf my 
# mkdir my
# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-0.5B --debug-mode output_hidden --gpu-batch-size 1 --num-gpu-batches 1 --gen-len 1

#### basic test
# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-0.5B  


# model=Qwen/Qwen2-0.5B
# model=Qwen/Qwen2-0.5B-Instruct 
# model=Qwen/Qwen2-7B
model="/home/llmserver/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B/snapshots/91d2aff3f957f99e4c74c962f2f408dcc88a18d8"
# model=/home/llmserver/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e
gpu_bs=8
gen_len=32
# python -m flexllmgen.flex_qwen2 \
# 	--model $model \
# 	--gpu-batch-size $gpu_bs \
# 	--num-gpu-batches 2 \
# 	--log-file log \
# 	--prompt-len 2100 \
# 	--gen-len 128

python -m flexllmgen.flex_qwen2 \
	--model $model \
	--gpu-batch-size $gpu_bs \
	--num-gpu-batches 2 \
	--prompt-len 2100 \
	--gen-len $gen_len \
	--sep-layer False \
	--log-file log \
	--sparse-mode block \
	--percent 100 0 0 100 100 0 \
	# --sparse-mode naive --attn-sparsity 0.1
	


# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-0.5B-Instruct --prompt-len 2048 --gen-len 256

# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-0.5B --attn-sparsity 0.1 --input-file prompt01.txt --gen-len 128
# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-7B 
# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-0.5B  --input-file prompt01.txt --gen-len 128
# python -m flexllmgen.flex_opt --model facebook/opt-125m --attn-sparsity 0.1
