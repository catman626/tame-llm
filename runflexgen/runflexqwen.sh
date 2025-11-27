# rm -rf my 
# mkdir my
# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-0.5B --debug-mode output_hidden --gpu-batch-size 1 --num-gpu-batches 1 --gen-len 1

#### basic test
# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-0.5B  


# model=Qwen/Qwen2-0.5B-Instruct 
model=Qwen/Qwen2-7B

python -m flexllmgen.flex_qwen2 --model $model --prompt-len 2048 --gen-len 256
# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-0.5B-Instruct --prompt-len 2048 --gen-len 256

# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-0.5B --attn-sparsity 0.1 --input-file prompt01.txt --gen-len 128
# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-7B 
# python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-0.5B  --input-file prompt01.txt --gen-len 128
# python -m flexllmgen.flex_opt --model facebook/opt-125m --attn-sparsity 0.1
