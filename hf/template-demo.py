from transformers import AutoModelForCausalLM, AutoTokenizer

def build_model_tokenizer(model_name):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def run_test(model, tokenizer):
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f" >>> model response: {response}")


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-1.7B"
    # model_name = "Qwen/Qwen2.5-7B-Instruct-1M"

    model, tokenizer = build_model_tokenizer(model_name)
    run_test(model, tokenizer)
