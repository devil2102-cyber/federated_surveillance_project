import cv2
import numpy as np
import torch
from torchvision import transforms

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image_path(self, image_path: str) -> torch.Tensor:
        """Reads and preprocesses an image from a path."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        return self.preprocess_image(img)
        
    def preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Preprocesses an OpenCV image (numpy array)."""
        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize
        img = cv2.resize(img, self.target_size)
        # Apply standard transforms (ToTensor + Normalize)
        return self.transform(img).unsqueeze(0) # Add batch dimension
