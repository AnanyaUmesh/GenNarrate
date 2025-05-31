import os
import re
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def clean_incomplete_ending(text):
    """
    Removes the last sentence if it doesn't end with a proper sentence-ending punctuation.
    """
    # Match sentences that end with '.', '!', or '?'
    sentences = re.findall(r'[^.!?]*[.!?]', text, re.DOTALL)

    # Join all complete sentences
    cleaned_text = ''.join(sentences).strip()

    return cleaned_text

def generate_story(prompt, max_tokens=500):
    """
    Generates a story based on the user's prompt and trims any incomplete ending.
    """
    try:
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": (
                    "You are a master storyteller AI. Your task is to craft a fully self-contained, immersive, and genre-appropriate story "
                    "based on the user’s prompt. The story must have a clear beginning, middle, and a satisfying, conclusive ending — all within the given token limit. "
                    "Avoid trailing off, incomplete endings, or tacked-on moral lessons. Ensure the story wraps up naturally and leaves the reader satisfied. "
                    "Do not generate placeholder text or continue beyond the ending. Write the complete story in one pass."
                )
            },
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.9,
        )

        story = response.choices[0].message.content.strip()

        # Clean up any incomplete ending
        return clean_incomplete_ending(story)

    except Exception as e:
        raise RuntimeError(f"Failed to generate story: {str(e)}")
