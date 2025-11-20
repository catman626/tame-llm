rm -rf my 
mkdir my
  
python -m flexllmgen.flex_qwen2 --model Qwen/Qwen2-0.5B --gen-len 3 --gpu-batch-size 1 --num-gpu-batches 1 --debug-mode basic