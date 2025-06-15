
from PIL import Image
import numpy as np

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((32, 32)).convert('L')
    return np.array(image).reshape(1, -1)
