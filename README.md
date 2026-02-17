# Emotion Detector

Tiny Python utility that detects emotions in **English text** using a Hugging Face Transformers pipeline.

- **Model**: `j-hartmann/emotion-english-distilroberta-base`
- **Task**: `text-classification` (returns all emotion scores via `top_k=None`)

## Setup

Create/activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install transformers torch
```

Notes:
- The **first run downloads** the model weights (requires internet).
- If you’re on Apple Silicon, `torch` wheels are supported; if install issues occur, use the official PyTorch install command for your platform.

## Usage

The core function is `emotion_detector(text)` in `emotion_detection.py`.

### Call from Python

```python
from emotion_detection import emotion_detector

print(emotion_detector("I’m really excited about this!"))
```

### One-liner

```bash
python -c "from emotion_detection import emotion_detector; print(emotion_detector('I’m really excited about this!'))"
```

## Output

For non-empty input, the function returns a dictionary mapping emotion labels to scores (rounded to 2 decimals), for example:

```python
{
  "joy": 0.87,
  "anger": 0.02,
  "sadness": 0.03
}
```

For blank input (`""` / `None`), it returns:

```python
({"error": "Input cannot be blank"}, 400)
```

## Project structure

- `emotion_detection.py`: Loads the model pipeline and exposes `emotion_detector(text)`.

## Model / credits

- Hugging Face model: `j-hartmann/emotion-english-distilroberta-base`
- Transformers library: `transformers`

