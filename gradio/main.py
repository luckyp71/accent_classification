import gradio as gr
import requests

def classify_accent(video_url):
    api_url = "http://localhost:8000/accent/classify"
    payload = {"url": video_url}

    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            result = response.json()
            return f"Prediction: {result['data']['prediction']['class']} with Confidence Score: {result['data']['prediction']['probability']:.2f})"
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Request failed: {e}"

gr.Interface(
    fn=classify_accent,
    inputs=gr.Textbox(label="Video URL"),
    outputs=gr.Textbox(label="Result"),
    title="Accent Classification",
    description="Enter a video URL to classify its accent."
).launch()