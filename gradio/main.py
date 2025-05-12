import gradio as gr
import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import requests
import tempfile

model = Wav2Vec2ForSequenceClassification.from_pretrained("luckyp71/accent-classifier-model")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("luckyp71/accent-classifier-model")

def classify_accent(video_url):
    try:
        # Download video
        response = requests.get(video_url)
        if response.status_code != 200:
            return f"Failed to download video: {response.status_code}"

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
            tmp_vid.write(response.content)
            tmp_vid.flush()

            # Extract audio (make sure ffmpeg is installed)
            waveform, sample_rate = torchaudio.load(tmp_vid.name)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

        inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()

        labels = model.config.id2label
        prediction = labels[pred_idx]
        confidence = probs[pred_idx].item()

        return f"Prediction: {prediction}, Confidence: {confidence:.2f}"

    except Exception as e:
        return f"Error during processing: {e}"


gr.Interface(
    fn=classify_accent,
    inputs=gr.Textbox(label="Video URL"),
    outputs=gr.Textbox(label="Prediction Result"),
    title="Accent Classification",
    description="Paste a video URL (e.g., Loom or direct MP4 link) to detect English accent."
).launch()