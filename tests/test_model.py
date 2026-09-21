import sys
import os
import numpy as np
import joblib

# Add project root to path so we can import model_classes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.model_classes import LogisticRegression, RidgePenalty, NoPenalty

# Load the saved model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
model = joblib.load(MODEL_PATH)

def test_model_input():
    """Test that the model accepts the expected input shape (46 features including intercept)"""
    sample = np.random.rand(1, 46)  # single sample, 46 features
    prediction = model.predict(sample)
    assert prediction is not None, "Model returned None for valid input"

def test_model_output_shape():
    """Test that model output has expected shape and valid class values"""
    samples = np.random.rand(10, 46)  # 10 samples
    predictions = model.predict(samples)
    assert predictions.shape == (10,), f"Expected shape (10,), got {predictions.shape}"
    assert all(p in [0, 1, 2, 3] for p in predictions), "Predictions should be in {0, 1, 2, 3}"