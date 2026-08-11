import requests
import torch
import torch.optim as optim
import torch.nn.functional as F
import io
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from client_edge.models.spatiotemporal_gnn import SpatioTemporalGNN
from shared.crypto_utils import generate_key_from_password, encrypt_data
from shared.cia_compliance_logger import cia_logger

class FederatedClient:
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url
        self.model = SpatioTemporalGNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        # Normally password would be securely loaded from an environment variable or local config
        self.key = generate_key_from_password("super_secret_federated_password")

    def local_train_gnn(self, data, epochs=5):
        """Simulate local training on graph data."""
        self.model.train()
        
        # Use the 3rd feature (infection status) as the target label to learn
        target = data.x[:, 2].long()
        
        losses = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        
        # Get final predictions after training
        self.model.eval()
        with torch.no_grad():
            final_out = self.model(data)
            # Softmax to get probability of being infected (class 1)
            predictions = F.softmax(final_out, dim=1)[:, 1].numpy()
            
        print(f"Local training complete. Final loss: {losses[-1]:.4f}")
        return losses, predictions

    def send_weights(self):
        """Encrypts local weights and sends to server."""
        state_dict = self.model.state_dict()
        
        # Serialize to bytes
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        raw_bytes = buffer.getvalue()
        
        # Encrypt
        encrypted_bytes = encrypt_data(raw_bytes, self.key)
        
        cia_logger.log_event("SENDING_WEIGHTS", "local", "START", f"Sending {len(encrypted_bytes)} bytes")
        
        try:
            response = requests.post(
                f"{self.server_url}/submit_weights", 
                data=encrypted_bytes, 
                headers={'Content-Type': 'application/octet-stream'}
            )
            response.raise_for_status()
            cia_logger.log_event("SENDING_WEIGHTS", "local", "SUCCESS", response.json().get('message', ''))
            print("Successfully submitted weights to aggregator.")
        except Exception as e:
            cia_logger.log_event("SENDING_WEIGHTS", "local", "FAILED", str(e))
            print(f"Failed to submit weights: {e}")
