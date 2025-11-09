from huggingface_hub import snapshot_download


model_name = "facebook/opt-125m"  
path = snapshot_download(model_name)
print(f" >>> cached path")