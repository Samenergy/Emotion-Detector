# emotion_detection.py
from transformers import pipeline

# Load model
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None  # IMPORTANT: returns all emotions
)

def emotion_detector(text):
    if not text:
        return {"error": "Input cannot be blank"}, 400

    results = emotion_classifier(text)

    # Handle output structure safely
    if isinstance(results, list) and len(results) > 0:
        results = results[0]  # unwrap batch

    # Convert to dictionary
    formatted = {}
    for item in results:
        label = item['label'].lower()
        score = round(item['score'], 2)
        formatted[label] = score

    return formatted
