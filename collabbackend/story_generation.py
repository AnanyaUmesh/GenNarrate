import os
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def is_story_incomplete(text: str) -> bool:
    # Naive heuristic: check for abrupt endings
    incomplete_endings = (
        text.endswith("...") or
        text.endswith(",") or
        text.endswith("and") or
        text.endswith("but") or
        not text.strip().endswith(('.', '!', '?'))
    )
    return incomplete_endings

def generate_story(prompt, max_tokens=500):
    try:
        initial_tokens = max_tokens - 50

        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": (
                    "You are an expert storyteller AI. You write immersive, creative, and genre-appropriate stories "
                    "based on the given prompt. Your task is to generate a complete, engaging story within the specified token limit. "
                    "Make sure the story follows the specified genre, has a beginning, middle, and end, and feels satisfying and imaginative."
                )
            },
            {"role": "user", "content": prompt}
        ]

        # First pass (reserve 50 tokens)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=initial_tokens,
            temperature=0.9,
        )
        partial_story = response.choices[0].message.content.strip()

        # Check if incomplete
        if is_story_incomplete(partial_story):
            # Continue story
            continuation_prompt = f"Continue the following story to a satisfying conclusion:\n\n{partial_story}"

            continuation_messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": "You are a story-finishing AI. Your job is to conclude incomplete stories meaningfully."},
                {"role": "user", "content": continuation_prompt}
            ]

            continuation_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=continuation_messages,
                max_tokens=50,
                temperature=0.9,
            )

            final_part = continuation_response.choices[0].message.content.strip()
            full_story = partial_story + " " + final_part
            return full_story.strip()

        else:
            return partial_story

    except Exception as e:
        raise RuntimeError(f"Failed to generate story: {str(e)}")
