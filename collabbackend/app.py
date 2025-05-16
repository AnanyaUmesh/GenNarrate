from flask import Flask, request, jsonify, send_from_directory
from story_generation import generate_story
from image_generation import generate_images_from_story
from audio_generation import generate_audio_from_story
from pdf_generation import create_pdf
from flask_cors import CORS
import os
import sys

app = Flask(__name__)
CORS(app)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/generate_story', methods=['POST'])
def generate_story_endpoint():
    try:
        data = request.json
        prompt = data.get("prompt")
        genre = data.get("genre")
        length = data.get("length")

        full_prompt = f"Genre: {genre}. Prompt: {prompt}"
        story = generate_story(full_prompt, length)

        with open(os.path.join(OUTPUT_DIR, "story.txt"), "w") as f:
            f.write(story)

        return jsonify({"story": story})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_images', methods=['POST'])
def generate_images_endpoint():
    try:
        data = request.json
        story = data.get("story")
        num_images = int(data.get("num_images", 1))
        image_urls = generate_images_from_story(story, num_images)
        return jsonify({"images": image_urls})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_audio', methods=['POST'])
def generate_audio_endpoint():
    try:
        data = request.json
        story = data.get("story")
        voice = data.get("voice")
        music_prompt = data.get("music_prompt")

        audio_path = generate_audio_from_story(story, voice, music_prompt)
        filename = os.path.basename(audio_path)
        return jsonify({"audio": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download_audio')
def download_audio():
    filename = request.args.get("file")
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    try:
        data = request.json
        story = data.get("story")
        image_urls = data.get("images", [])
        pdf_path = create_pdf(story, image_urls)
        return jsonify({"pdf": os.path.basename(pdf_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

@app.route("/")
def home():
    return "Hello from GenNarrate"

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    app.run(port = port)
