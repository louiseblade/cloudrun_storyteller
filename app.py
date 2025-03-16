from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gtts import gTTS
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
CORS(app)

# Model path in the container (configurable via environment variable)
MODEL_PATH = os.getenv("MODEL_PATH", "/app/goat_70b_local")

# Load tokenizer and model with PyTorch and transformers
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,  # Reduces memory usage
        device_map="auto",           # Distributes across available GPUs/CPU
        offload_folder="offload"     # Folder for offloaded weights
    )
except Exception as e:
    app.logger.error(f"Failed to load model: {e}")
    raise

# Audio directory
AUDIO_DIR = os.path.join(os.getcwd(), "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

@app.route('/')
def index():
    return "GOAT-70B story generator online!"

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

@app.route('/generate-story', methods=['POST'])
def generate_story():
    data = request.json or {}
    prompt = data.get('prompt', 'Once upon a time')

    input_text = f"Write a short story about: {prompt}"
    try:
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=400, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return jsonify({"error": f"Failed to generate story: {str(e)}"}), 500

    # Generate TTS audio
    audio_file = os.path.join(AUDIO_DIR, "story.mp3")
    try:
        tts = gTTS(text=generated_text, lang='en', slow=False)
        tts.save(audio_file)
    except Exception as e:
        return jsonify({"error": f"Failed to generate audio: {str(e)}"}), 500

    # Audio URL configuration
    CLOUD_RUN_SERVICE_URL = os.getenv("CLOUD_RUN_SERVICE_URL", request.host_url.rstrip('/'))
    audio_url = f"{CLOUD_RUN_SERVICE_URL}/audio/story.mp3"

    return jsonify({
        "story": generated_text,
        "audio": audio_url
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)