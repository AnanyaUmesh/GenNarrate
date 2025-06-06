from flask import Flask, request, jsonify, send_from_directory, send_file
from google.colab import files
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
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")        # Your audio files are saved here
PDF_DIR = os.path.join(OUTPUT_DIR, "pdf")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)  # Make sure PDF output folder exists

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

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf_endpoint():
    try:
        data = request.json
        story = data.get("story")
        image_urls = data.get("images", [])

        pdf_path = create_pdf(story, image_urls)
        filename = os.path.basename(pdf_path)

        return jsonify({"pdf": filename})  # just return file name
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/trigger_pdf_download', methods=['POST'])
def trigger_pdf_download():
    try:
        data = request.json
        filename = data.get("filename")
        full_path = os.path.join(PDF_DIR, filename)
        if not os.path.isfile(full_path):
            return "File not found", 404
        files.download(full_path)  # this triggers the Colab system download
        return "Success", 200
    except Exception as e:
        return str(e), 500

@app.route('/trigger_audio_download', methods=['POST'])
def trigger_audio_download():
    try:
        data = request.json
        filename = data.get("filename")
        full_path = os.path.join(AUDIO_DIR, filename)
        if not os.path.isfile(full_path):
            return "File not found", 404
        files.download(full_path)
        return "Success", 200
    except Exception as e:
        return str(e), 500
        
@app.route("/")
def home():
    return "Hello from GenNarrate"

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    app.run(port=port)
