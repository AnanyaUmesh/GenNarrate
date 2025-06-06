from fpdf import FPDF
import requests
from PIL import Image
from io import BytesIO
import os
import unicodedata

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    return unicodedata.normalize("NFKD", text).encode("latin-1", "ignore").decode("latin-1")

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Generated Story", ln=True, align="C")

    def add_story_text(self, text):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, clean_text(text))  # <--- cleaned text here

    def add_image(self, image_url, idx):
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            temp_path = os.path.join(OUTPUT_DIR, f"temp_image_{idx}.jpg")
            image.save(temp_path)
            self.image(temp_path, w=150)
        except Exception as e:
            print(f"Failed to load image into PDF: {e}")

def create_pdf(story_text, image_urls):
    pdf = PDF()
    pdf.add_page()
    pdf.add_story_text(story_text)

    for idx, url in enumerate(image_urls):
        pdf.add_page()
        pdf.add_image(url, idx)

    output_path = os.path.join(OUTPUT_DIR, "story_output.pdf")
    pdf.output(output_path)
    return output_path
