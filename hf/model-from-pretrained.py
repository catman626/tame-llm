from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


if __name__ == "__main__":
    model_name =  "Qwen/Qwen2.5-7B"
#     model_name = "/home/llmserver/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B/snapshots/91d2aff3f957f99e4c74c962f2f408dcc88a18d8"
        

    model = AutoModelForCausalLM.from_pretrained( model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, 
            device_map="auto",
            #attn_implementation="flash_attention_2"
    )
    

    inputs = ["Paris is the capital city of"]
    input_ids = tokenizer(inputs).input_ids # return [int, ], need convert to tensor
    input_ids = torch.tensor(input_ids)     

    # generate step
    outputs = model.generate(input_ids)

    outputs = tokenizer.batch_decode(outputs)

    print(outputs)


