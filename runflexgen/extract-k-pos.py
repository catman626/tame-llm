import torch
import sys

# (b, head, s, h)
save_file  = sys.argv[1]
k = torch.load(save_file)
k_pos = k[:, :, -1:]
others = k[:, :, :-2]

torch.save(k_pos, "extracted")
torch.save(others, "kcache")