# DeepTrace: Federated Graph Learning for Disease Surveillance

## Overview
DeepTrace is a proactive public health architecture designed to predict and neutralize transmission vectors before outbreaks occur. By treating disease transmission as a complex network problem, this system utilizes Graph Neural Networks (GNNs) and federated learning to achieve high detection accuracy without centralizing sensitive patient data. It acts as an Intrusion Prevention System (IPS) for public health, identifying anomalies at the edge and algorithmically isolating "bridge nodes" to halt transmission chains.

## System Architecture
The platform is divided into two distinct environments to guarantee data sovereignty and zero-trust security:
1. **Edge Client (Local):** Deployed on clinical infrastructure or mobile devices. It handles localized data ingestion, computer vision diagnostics via CNNs, and local graph construction.
2. **Aggregator Server (Global):** A centralized server that receives encrypted mathematical model weights—never raw data—to update the global predictive model.

## Core Engineering Concepts

### 1. Data Structures & Graph Traversal
The transmission network is modeled strictly using graph data structures.
*   **Adjacency Lists:** Patient interactions and physical proximity events are stored locally as adjacency lists. Nodes represent individuals, and weighted edges represent the statistical probability of transmission.
*   **BFS and DFS Integration:** The `deep_trace_mle.py` module executes Breadth-First Search (BFS) to map immediate local exposure clusters, while Depth-First Search (DFS) runs backward through historical interaction data to pinpoint the original source node (Patient Zero).

### 2. Cybersecurity & The CIA Triad
To ensure public trust and legal compliance, the architecture is strictly bound by the Information Security Management principles of the CIA triad:
*   **Confidentiality:** Raw diagnostic data and interaction graphs never leave the edge device. Only encrypted GNN weights are transmitted over the network.
*   **Integrity:** Incoming weights from local nodes are authenticated at the aggregator server to prevent data poisoning attacks against the global model.
*   **Availability:** The federated, decentralized nature of the network ensures that even if several local clinic nodes go offline, the global surveillance system remains fully operational and highly available.
*   **System Hardening:** All client-to-server communication is restricted to secure protocols (SSH/TLS) with strict firewall rules blocking unauthorized port access.

## Project Structure
\`\`\`text
federated_surveillance_project/
├── server/                              # Central Aggregator Environment
│   ├── aggregator.py                    
│   ├── server_config.yaml               
│   └── global_model_state.pt            
├── client_edge/                         # Edge Client Environment
│   ├── data_structures/
│   │   ├── graph_builder.py             
│   │   └── node_features.py             
│   ├── diagnostics/
│   │   ├── edge_cnn_classifier.py       
│   │   └── image_preprocessor.py        
│   ├── models/
│   │   ├── spatiotemporal_gnn.py        
│   │   └── federated_client.py          
│   └── tracing_algorithms/
│       ├── deep_trace_mle.py            
│       ├── bfs_traversal.py             
│       └── dfs_traversal.py             
├── shared/                              
│   ├── crypto_utils.py                  
│   └── cia_compliance_logger.py         
├── requirements.txt                     
└── README.md                            
\`\`\`

## Installation & Deployment

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/your-org/DeepTrace.git
   cd DeepTrace
   \`\`\`

2. **Install core dependencies:**
   The environment requires PyTorch for neural network execution and NetworkX for graph pathfinding.
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Deploy the Aggregator Server:**
   Initialize the central server to begin listening for incoming encrypted weights.
   \`\`\`bash
   uvicorn server.aggregator:app --host 0.0.0.0 --port 8000
   \`\`\`

4. **Initialize an Edge Client:**
   Launch the local processing unit to begin constructing the adjacency list and training the local GNN.
   \`\`\`bash
   python client_edge/models/federated_client.py
   \`\`\`
