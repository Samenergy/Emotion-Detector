# emotion_detection.py
from transformers import pipeline

# Load Hugging Face emotion detection model
emotion_classifier = pipeline("text-classification", 
                              model="j-hartmann/emotion-english-distilroberta-base", 
                              return_all_scores=True)

def emotion_detector(text):
    if not text:
        return {"error": "Input cannot be blank"}, 400
    results = emotion_classifier(text)[0]  # get first batch
    # Format output
    formatted = {item['label'].lower(): round(item['score'], 2) for item in results}
    return formatted
