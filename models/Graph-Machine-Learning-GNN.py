!pip install torch torchvision torch-geometric

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# ตัวอย่าง graph: 4 nodes, 2 edges
edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long).t().contiguous()
x = torch.randn((4, 16))  # node features
data = Data(x=x, edge_index=edge_index)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN(16, 2)
out = model(data.x, data.edge_index)
print(out.shape)  # torch.Size([4, 2])
