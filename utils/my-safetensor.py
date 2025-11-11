from safetensors.torch import load_file
import sys


path = sys.argv[1]
data = load_file(path)

for n in data.keys():
    print(f"{n}, {data[n].shape}")
