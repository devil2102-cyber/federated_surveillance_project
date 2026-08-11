import torch
import yaml
import io
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import sys
import traceback

# Ensure shared directory is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.crypto_utils import generate_key_from_password, decrypt_data
from shared.cia_compliance_logger import cia_logger

app = FastAPI()

def load_config():
    with open(os.path.join(os.path.dirname(__file__), 'server_config.yaml'), 'r') as f:
        return yaml.safe_load(f)

config = load_config()
SERVER_KEY = generate_key_from_password(config['server']['secret_password'])
GLOBAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), config['model']['global_state_path'])

def average_weights(w1, w2, weight_ratio=0.5):
    """Average two state dicts."""
    w_avg = {}
    for k in w1.keys():
        if isinstance(w1[k], torch.Tensor) and w1[k].is_floating_point():
            w_avg[k] = w1[k] * weight_ratio + w2[k] * (1 - weight_ratio)
        else:
            # For non-floating point (e.g. num_batches_tracked), just take the newer one
            w_avg[k] = w2[k]
    return w_avg

@app.post("/submit_weights")
async def submit_weights(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    try:
        # Expecting raw bytes in the request body
        encrypted_data = await request.body()
        
        if not encrypted_data:
            raise HTTPException(status_code=400, detail="Empty payload")

        cia_logger.log_event("RECEIVED_WEIGHTS", client_ip, "SUCCESS", f"Received {len(encrypted_data)} bytes")

        # Decrypt
        decrypted_data = decrypt_data(encrypted_data, SERVER_KEY)
        
        # Load state dict
        buffer = io.BytesIO(decrypted_data)
        client_state_dict = torch.load(buffer, weights_only=True)
        
        cia_logger.log_event("DECRYPTED_WEIGHTS", client_ip, "SUCCESS", "Successfully decrypted and loaded state dict")
        
        # Average with global state
        if os.path.exists(GLOBAL_MODEL_PATH):
            global_state_dict = torch.load(GLOBAL_MODEL_PATH, weights_only=True)
            new_global_state = average_weights(global_state_dict, client_state_dict, weight_ratio=0.5)
        else:
            new_global_state = client_state_dict
            
        # Save back
        torch.save(new_global_state, GLOBAL_MODEL_PATH)
        
        cia_logger.log_event("AGGREGATED_WEIGHTS", client_ip, "SUCCESS", "Averaged and saved global state")
        
        return JSONResponse(content={"status": "success", "message": "Weights aggregated successfully."})
        
    except Exception as e:
        error_details = traceback.format_exc()
        cia_logger.log_event("ERROR_SUBMITTING_WEIGHTS", client_ip, "FAILED", error_details)
        raise HTTPException(status_code=500, detail=str(e) if str(e) else repr(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config['server']['host'], port=config['server']['port'])
