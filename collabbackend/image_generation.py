import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_key_moments(story: str, n: int):
    try:
        prompt = f"Extract {n} key visual moments from this story for image generation:\n{story}"

        response = client.chat.completions.create(
            model="gpt-4",
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
                size="512x512",
                model="dall-e-3"  # or "dall-e-2" if DALL·E 3 access isn't enabled
            )
            images.append(image_resp.data[0].url)

        return images
    
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {str(e)}")
