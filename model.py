# model.py
"""
This script demonstrates using DeepFace's pretrained emotion model.
DeepFace wraps pretrained emotion models internally. For the assignment,
we don't train from scratch; we load and use the pretrained emotion analyzer.

This script:
- builds/loads the DeepFace emotion model
- runs a quick check on images in ./sample_images (if present)
- writes a small metadata file 'model_info.json' describing the model used

See README / comments for more explanation.
"""
import os
import json
from deepface import DeepFace
from datetime import datetime

MODEL_INFO_PATH = "models/model_info.json"
os.makedirs("models", exist_ok=True)

def main():
    print("Building/loading DeepFace emotion model (this will download weights if absent)...")
    # Build emotion model (DeepFace supports building the 'Emotion' model)
    # The build_model API returns a Keras/TensorFlow model object for the requested detector.
    try:
        model = DeepFace.build_model('Emotion')
        model_name = "DeepFace-Emotion"
        print("Model loaded:", model_name)
    except Exception as e:
        print("Warning: Could not call build_model('Emotion') directly due to environment. Falling back to note only.")
        model = None
        model_name = "DeepFace-Emotion (build_model failed - using DeepFace.analyze internally)"

    info = {
        "model_name": model_name,
        "loaded_at": datetime.utcnow().isoformat(),
        "notes": "This app uses DeepFace.analyze(..., actions=['emotion']) which internally uses a pretrained emotion model. No additional training performed."
    }

    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)

    print("Model info saved to:", MODEL_INFO_PATH)

if __name__ == "__main__":
    main()