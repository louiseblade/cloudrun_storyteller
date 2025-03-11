from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gtts import gTTS
import os
from llama_cpp import Llama

app = Flask(__name__)
CORS(app)

# Define the mounted model path (adjust if needed based on your mount)
MODEL_PATH = os.path.join("/app", "model", "goat-70b-storytelling.Q4_K_M.gguf")

# Initialize model with GPU support, using the mounted model path
llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=29)

# Audio directory
AUDIO_DIR = os.path.join(os.getcwd(), "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

@app.route('/')
def index():
    return "GOAT-70B story generator online!"

# Serve audio files statically
@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

@app.route('/generate-story', methods=['POST'])
def generate_story():
    data = request.json or {}
    prompt = data.get('prompt', 'Once upon a time')

    input_text = f"Write a short story about: {prompt}"

    # Reuse the GPU-enabled model instance
    output = llm(prompt=input_text,
                 max_tokens=400,
                 temperature=0.7,
                 top_k=50,
                 repeat_penalty=1.2,
                 )

    generated_text = output['choices'][0]['text'].strip()

    # Generate TTS audio
    audio_file = os.path.join(AUDIO_DIR, "story.mp3")
    tts = gTTS(text=generated_text, lang='en', slow=False)
    tts.save(audio_file)

    # Audio URL configuration
    CLOUD_RUN_SERVICE_URL = os.getenv("CLOUD_RUN_SERVICE_URL", "http://127.0.0.1:8080")
    audio_url = f"{CLOUD_RUN_SERVICE_URL}/audio/story.mp3"

    return jsonify({
        "story": generated_text,
        "audio": audio_url
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)