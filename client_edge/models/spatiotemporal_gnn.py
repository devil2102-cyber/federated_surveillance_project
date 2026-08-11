import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SpatioTemporalGNN(torch.nn.Module):
    """A GNN for processing the surveillance graph (spatiotemporal data)."""
    def __init__(self, num_node_features=3, hidden_channels=16, num_classes=2):
        super(SpatioTemporalGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        if edge_weight is not None and edge_weight.numel() > 0:
            edge_weight = edge_weight.squeeze() # Remove extra dims if needed
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
        else:
            x = self.conv1(x, edge_index)
            
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        if edge_weight is not None and edge_weight.numel() > 0:
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
        else:
            x = self.conv2(x, edge_index)
            
        x = F.relu(x)

        # Output predictions per node (e.g. risk of infection)
        return self.lin(x)
