
 # model.py
import tensorflow as tf
from utils import preprocess_image

MODEL_PATH = "models/Cancer30epoch.h5"

# Load CNN model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict_risk(image_bytes):
    """
    Runs CNN prediction on uploaded image bytes.
    Returns JSON-safe values.
    """

    image = preprocess_image(image_bytes)

    prediction = model.predict(image)[0][0]

    confidence = float(prediction)
    risk = "High Risk" if confidence >= 0.5 else "Low Risk"

    return {
        "risk": risk,
        "confidence": confidence
    }
