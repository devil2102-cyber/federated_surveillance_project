import sys
import os
import time

# Add root to pythonpath
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from client_edge.data_structures.node_features import PatientNodeFeatures, LocationNodeFeatures
from client_edge.data_structures.graph_builder import GraphBuilder
from client_edge.models.federated_client import FederatedClient

def run_test():
    print("Building surveillance graph...")
    builder = GraphBuilder()
    
    # Add patients
    p1 = PatientNodeFeatures(patient_id="P001", age=45, symptom_severity=0.8, is_infected=True)
    p2 = PatientNodeFeatures(patient_id="P002", age=30, symptom_severity=0.0, is_infected=False)
    builder.add_patient(p1)
    builder.add_patient(p2)
    
    # Add location
    l1 = LocationNodeFeatures(location_id="L100", capacity=50, current_occupancy=20, ventilation_score=0.4)
    builder.add_location(l1)
    
    # Add interactions
    builder.add_interaction("P001", "L100", duration_minutes=120)
    builder.add_interaction("P002", "L100", duration_minutes=60)
    
    # Convert to PyTorch Geometric data
    graph_data = builder.to_pyg_data()
    print(f"Graph Data generated: {graph_data}")
    
    print("\nStarting local federated training...")
    client = FederatedClient()
    client.local_train_gnn(graph_data, epochs=3)
    
    print("\nSending weights to aggregator...")
    # NOTE: Ensure the FastAPI server is running before executing this script!
    try:
        client.send_weights()
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        print("Please start the server first by running: python server/aggregator.py")

if __name__ == "__main__":
    run_test()
