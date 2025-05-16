import pickle
import os
import re
import numpy as np

# Load saved model and vectorizer
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def clean_text(text):
    """Clean and preprocess the text for better prediction"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_news(text):
    if not text.strip():
        return "⚠️ Empty input", 0.0

    # Clean the text before prediction
    cleaned_text = clean_text(text)
    
    vec = tfidf.transform([cleaned_text])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    
    # Get confidence scores for both classes
    fake_confidence = proba[0] * 100  # Probability of FAKE (assuming index 0 is FAKE)
    real_confidence = proba[1] * 100  # Probability of REAL (assuming index 1 is REAL)
    
    # Apply a more balanced threshold instead of just using argmax
    # If the difference between probabilities is small, be more cautious
    if real_confidence > fake_confidence:
        if real_confidence - fake_confidence < 15:  # If difference is less than 15%, be cautious
            label = "UNCERTAIN" if real_confidence < 65 else "LIKELY REAL"
            confidence = real_confidence
        else:
            label = "REAL"
            confidence = real_confidence
    else:
        if fake_confidence - real_confidence < 15:  # If difference is less than 15%, be cautious
            label = "UNCERTAIN" if fake_confidence < 65 else "LIKELY FAKE"
            confidence = fake_confidence
        else:
            label = "FAKE"
            confidence = fake_confidence
            
    # Return additional information for debugging
    return label, confidence, {"fake_conf": fake_confidence, "real_conf": real_confidence}
