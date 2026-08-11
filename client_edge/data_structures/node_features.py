from dataclasses import dataclass
import torch

@dataclass
class PatientNodeFeatures:
    """Represents a patient in the surveillance graph."""
    patient_id: str
    age: int
    symptom_severity: float # 0.0 to 1.0
    is_infected: bool
    
    def to_tensor(self) -> torch.Tensor:
        # Example feature vector representation
        return torch.tensor([float(self.age) / 100.0, self.symptom_severity, float(self.is_infected)], dtype=torch.float32)

@dataclass
class LocationNodeFeatures:
    """Represents a location where transmission might occur."""
    location_id: str
    capacity: int
    current_occupancy: int
    ventilation_score: float # 0.0 to 1.0
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([float(self.capacity) / 1000.0, float(self.current_occupancy) / 1000.0, self.ventilation_score], dtype=torch.float32)
