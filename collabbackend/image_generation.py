import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_key_moments(story, n):
    try:
        prompt = f"Extract {n} key visual moments from this story for image generation:\n{story}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        moments = response.choices[0].message['content'].split('\n')
        return [m.strip('- ').strip() for m in moments if m.strip()]
    except Exception as e:
        raise RuntimeError(f"Error extracting key moments: {str(e)}")

def generate_images_from_story(story, num_images=1):
    try:
        prompts = extract_key_moments(story, num_images)
        images = []
        for p in prompts[:num_images]:
            image_resp = openai.Image.create(
                prompt=p,
                n=1,
                size="512x512"
            )
            images.append(image_resp['data'][0]['url'])  # URL format
        return images
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {str(e)}")
