
# utils.py
import cv2
import numpy as np

IMG_SIZE = (224, 224)  # (width, height)

def preprocess_image(image_bytes):
    """
    Convert uploaded image bytes into a CNN-ready tensor
    Shape output: (1, 224, 224, 3)
    """

    # Convert bytes → numpy array
    image = np.frombuffer(image_bytes, np.uint8)

    # Decode image
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image file")

    # BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize correctly
    image = cv2.resize(image, IMG_SIZE)

    # Normalize
    image = image.astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image
