import networkx as nx
import torch
from torch_geometric.data import Data
from .node_features import PatientNodeFeatures, LocationNodeFeatures

class GraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()
        self.node_mapping = {} # Maps logical IDs to integer indices for PyG
        self.current_idx = 0

    def add_patient(self, patient: PatientNodeFeatures):
        if patient.patient_id not in self.node_mapping:
            self.node_mapping[patient.patient_id] = self.current_idx
            self.graph.add_node(self.current_idx, features=patient.to_tensor(), type='patient')
            self.current_idx += 1

    def add_location(self, location: LocationNodeFeatures):
        if location.location_id not in self.node_mapping:
            self.node_mapping[location.location_id] = self.current_idx
            self.graph.add_node(self.current_idx, features=location.to_tensor(), type='location')
            self.current_idx += 1

    def add_interaction(self, id1: str, id2: str, duration_minutes: float):
        """Adds an edge between two entities (e.g., patient-patient or patient-location)."""
        idx1 = self.node_mapping.get(id1)
        idx2 = self.node_mapping.get(id2)
        if idx1 is not None and idx2 is not None:
            # Edge weight can be derived from duration
            weight = min(duration_minutes / 60.0, 1.0)
            self.graph.add_edge(idx1, idx2, weight=weight)

    def to_pyg_data(self) -> Data:
        """Converts the NetworkX graph to a PyTorch Geometric Data object."""
        if len(self.graph.nodes) == 0:
            return Data(x=torch.empty((0, 3)), edge_index=torch.empty((2, 0), dtype=torch.long))

        # Assumes all node features have the same dimensionality (e.g., 3) for simplicity
        x = [self.graph.nodes[i]['features'] for i in range(self.current_idx)]
        x = torch.stack(x)
        
        edges = list(self.graph.edges(data=True))
        if not edges:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)
        else:
            edge_index = torch.tensor([[e[0], e[1]] for e in edges], dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor([[e[2].get('weight', 1.0)] for e in edges], dtype=torch.float32)
            
            # Make undirected (add reverse edges)
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
