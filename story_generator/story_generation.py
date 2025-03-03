
# Import necessary libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model name: You must have access granted for this model
model_name = "meta-llama/Llama-2-7b-chat-hf"  # or Llama-2-13b-chat-hf

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16  # Use FP16 for performance
)

# Updated story generation function

def generate_story_with_completion(prompt, max_length, max_attempts=3):
    """
    Generates a coherent story with proper completion by avoiding infinite recursion.

    Parameters:
        prompt (str): The story prompt.
        max_length (int): The maximum tokens for each generation attempt.
        max_attempts (int): Maximum additional attempts to complete the story.

    Returns:
        str: The complete story.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Move to GPU
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=0.7,  # Controls randomness
        top_p=0.9,        # Nucleus sampling
        repetition_penalty=1.1  # Avoid repetitive text
    )
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Attempt to complete the story if it seems incomplete
    attempts = 0
    while not story.endswith((".", "!", "?")) and attempts < max_attempts:
        print("\nStory seems incomplete. Generating continuation...")
        continuation_prompt = story + " Finish the story in a single paragraph."
        inputs = tokenizer(continuation_prompt, return_tensors="pt").to("cuda")
        continuation = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,  # Small continuation length
            temperature=0.7,
            top_p=0.9
        )
        continuation_text = tokenizer.decode(continuation[0], skip_special_tokens=True)
        story += " " + continuation_text
        attempts += 1

    return story
