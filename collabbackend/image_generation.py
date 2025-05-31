import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_key_moments(story: str, n: int):
    try:
        prompt = (
            f"Read the following story and extract {n} visually rich moments. "
            "DO NOT include character names. Instead, describe their species (e.g., 'a clever fox', 'an old owl'). "
            "Make the prompts short, vivid, and descriptive enough to generate high-quality images. "
            "Only return image generation prompts, each on a new line:\n\n"
            f"{story}"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )

        content = response.choices[0].message.content
        moments = content.split('\n')
        return [m.strip('- ').strip() for m in moments if m.strip()]
    
    except Exception as e:
        raise RuntimeError(f"Error extracting key moments: {str(e)}")


def generate_images_from_story(story: str, num_images: int = 1):
    try:
        prompts = extract_key_moments(story, num_images)
        images = []

        for p in prompts[:num_images]:
            image_resp = client.images.generate(
                prompt=p,
                n=1,
                size="1024x1024",
                model="dall-e-3"  # or "dall-e-2" if needed
            )
            images.append(image_resp.data[0].url)

        return images
    
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {str(e)}")
