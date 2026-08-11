import logging
import json
from datetime import datetime
import os

class CIALogger:
    def __init__(self, name: str, log_file: str = "cia_audit.log"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        
        # Ensure the log file can be created if it's in a subdirectory
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_event(self, action: str, source_ip: str, status: str, details: str):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "action": action,
            "source_ip": source_ip,
            "status": status,
            "details": details
        }
        self.logger.info(json.dumps(log_entry))

# Global instance for easy importing
cia_logger = CIALogger("federated_surveillance")
