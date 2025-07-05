
import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops

for epoch in range(10):
    print(epoch)
