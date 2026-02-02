from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import sys

import argparse

def get_test_inputs(input_file):
    if input_file is None:
        return  [ "Paris is the capital city of "]
    inputs = [ open(input_file).read() ]
    return inputs

def run_model(model_name, inputs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained( model_name, attn_implementation="eager",)
    
    input_ids = tokenizer(inputs).input_ids # return [[int, ]], need convert to tensor
    input_ids = torch.tensor(input_ids)     

    outputs = model.generate(input_ids)
    outputs = tokenizer.batch_decode(outputs)

    for o in outputs:
        print(o)


if __name__ == "__main__":
    dump_dir = "dump"
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()
    
    inputs = get_test_inputs(args.input_file)

    run_model("Qwen/Qwen2.5-7B-Instruct", inputs)



    




