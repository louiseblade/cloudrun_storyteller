from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import os

app = Flask(__name__)
CORS(app)

device = torch.device("cpu")

# Get Cloud Run service URL from env variable
CLOUD_RUN_SERVICE_URL = os.getenv("CLOUD_RUN_SERVICE_URL", "http://127.0.0.1:8080")
IS_CLOUD_RUN = os.getenv("K_SERVICE") is not None  # Detect if running on Cloud Run

MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="./cache", force_download=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir="./cache", force_download=False)

model.eval()
model.to(device)

# Create the audio directory inside the container
AUDIO_DIR = os.path.join(os.getcwd(), "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

@app.route('/')
def index():
    return "Flan-T5 story generator online!"

# **Serve static audio files**
@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

@app.route('/generate-story', methods=['POST'])
def generate_story():
    data = request.json or {}
    prompt = data.get('prompt', 'Once upon a time')

    input_text = f"Write a short story about: {prompt}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=400,
        min_length=100,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        temperature=0.7,
        repetition_penalty=1.2
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Generate TTS and save inside the Docker container
    audio_file = os.path.join(AUDIO_DIR, "story.mp3")
    tts = gTTS(text=generated_text, lang='en', slow=False)
    tts.save(audio_file)

    # **Dynamically set audio URL**
    if IS_CLOUD_RUN:
        audio_url = f"{CLOUD_RUN_SERVICE_URL}/audio/story.mp3"
    else:
        audio_url = "http://127.0.0.1:8080/audio/story.mp3"  # When running locally

    return jsonify({
        "story": generated_text,
        "audio": audio_url
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
