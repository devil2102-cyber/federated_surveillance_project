import streamlit as st
import networkx as nx
import torch
import sys
import os
import time
from pyvis.network import Network
import streamlit.components.v1 as components

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from client_edge.data_structures.node_features import PatientNodeFeatures, LocationNodeFeatures
from client_edge.data_structures.graph_builder import GraphBuilder
from client_edge.models.federated_client import FederatedClient
from client_edge.tracing_algorithms.bfs_traversal import bfs_find_cluster
from client_edge.tracing_algorithms.deep_trace_mle import calculate_mle_source

st.set_page_config(page_title="Federated Surveillance Dashboard", layout="wide")
st.title("Federated Surveillance Dashboard 🌐🦠")
st.markdown("A federated graph learning system for pandemic tracking and privacy-preserving ML.")

@st.cache_resource
def build_graph():
    builder = GraphBuilder()
    
    # Generate dummy data
    p1 = PatientNodeFeatures(patient_id="Patient 1", age=45, symptom_severity=0.8, is_infected=True)
    p2 = PatientNodeFeatures(patient_id="Patient 2", age=30, symptom_severity=0.0, is_infected=False)
    p3 = PatientNodeFeatures(patient_id="Patient 3", age=60, symptom_severity=0.9, is_infected=True)
    builder.add_patient(p1)
    builder.add_patient(p2)
    builder.add_patient(p3)
    
    l1 = LocationNodeFeatures(location_id="Clinic A", capacity=50, current_occupancy=20, ventilation_score=0.4)
    l2 = LocationNodeFeatures(location_id="Supermarket B", capacity=200, current_occupancy=150, ventilation_score=0.8)
    builder.add_location(l1)
    builder.add_location(l2)
    
    builder.add_interaction("Patient 1", "Clinic A", duration_minutes=120)
    builder.add_interaction("Patient 2", "Clinic A", duration_minutes=60)
    builder.add_interaction("Patient 1", "Supermarket B", duration_minutes=30)
    builder.add_interaction("Patient 3", "Supermarket B", duration_minutes=45)
    
    return builder

builder = build_graph()
graph_data = builder.to_pyg_data()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Surveillance Interaction Graph")
    # Set up Pyvis Network
    net = Network(height="500px", width="100%", bgcolor="#1E1E1E", font_color="white")
    for node, data in builder.graph.nodes(data=True):
        color = "#ff4b4b" if data.get('type') == 'patient' else "#00b4d8"
        # Recover logical ID from builder mapping
        logical_id = [k for k, v in builder.node_mapping.items() if v == node][0]
        net.add_node(node, label=logical_id, color=color)
    
    for u, v, d in builder.graph.edges(data=True):
        net.add_edge(u, v, value=d.get('weight', 1.0))
        
    try:
        net.save_graph("graph.html")
        with open("graph.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=520)
    except Exception as e:
        st.error(f"Failed to render graph: {e}")

with col2:
    st.subheader("Federated Learning Controls")
    epochs = st.slider("Local Epochs", min_value=1, max_value=20, value=3)
    if st.button("Train Local GNN & Submit Weights"):
        with st.spinner("Training local model..."):
            client = FederatedClient()
            losses, predictions = client.local_train_gnn(graph_data, epochs=epochs)
            st.success("Local training complete.")
            
            st.markdown("##### 📉 Training Loss Curve")
            st.caption("Watch the model learn and decrease its error over time:")
            st.line_chart(losses)
            
            st.markdown("##### 🧠 Model Predictions")
            st.caption("The AI's predicted infection risk for each entity based on the graph:")
            
            cols = st.columns(3)
            col_idx = 0
            for node_idx, pred in enumerate(predictions):
                logical_id = [k for k, v in builder.node_mapping.items() if v == node_idx][0]
                node_type = builder.graph.nodes[node_idx]['type']
                if node_type == 'patient':
                    cols[col_idx % 3].metric(label=logical_id, value=f"{pred*100:.1f}%")
                    col_idx += 1
            
            with st.spinner("Encrypting and submitting weights..."):
                try:
                    client.send_weights()
                    st.success("Weights securely encrypted and submitted to Aggregator successfully!")
                except Exception as e:
                    st.error(f"Failed to submit: Is the server running? Error: {e}")
                    
    st.subheader("Tracing Algorithms")
    if st.button("Calculate MLE Outbreak Source"):
        infected = [i for i, d in builder.graph.nodes(data=True) if d['type'] == 'patient' and float(d['features'][2]) > 0.5]
        if infected:
            source = calculate_mle_source(builder.graph, infected)
            if source is not None:
                logical_id = [k for k, v in builder.node_mapping.items() if v == source][0]
                st.info(f"Most likely patient source (MLE): **{logical_id}**")
            else:
                st.warning("Could not trace a source.")
        else:
            st.warning("No infected patients found in graph.")
