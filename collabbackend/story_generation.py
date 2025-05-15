import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_story(prompt, max_tokens=500):
    try:
        system_prompt = (
            "You are an expert storyteller AI. You write immersive, creative, and genre-appropriate stories "
            "based on the given prompt. Your task is to generate a complete, engaging story within the specified token limit. "
            "Make sure the story follows the specified genre, has a beginning, middle, and end, and feels satisfying and imaginative."
        )

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.9,
        )

        return response.choices[0].message['content'].strip()

    except Exception as e:
        raise RuntimeError(f"Failed to generate story: {str(e)}")
