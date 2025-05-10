import os
from utils.util import get_direct_audio_url, extract_audio_from_url
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
# from dotenv import load_dotenv

# load_dotenv()

# model_name = os.getenv('accent_classifier')
model_name = 'luckyp71/accent-classifier-model'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict_audio_from_url(url):
    if "youtube.com" in url or "youtu.be" in url:
        media_url = get_direct_audio_url(url)
    else:
        media_url = url

    # Extract audio
    audio, sr = extract_audio_from_url(media_url, target_sr=16000)

    # Process input
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    # Get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()

    predicted_idx = probs.argmax()
    predicted_prob = probs[predicted_idx]
    predicted_label = model.config.id2label[predicted_idx]
    
    return {
        "class": predicted_label,
        "probability": float(predicted_prob)
    }