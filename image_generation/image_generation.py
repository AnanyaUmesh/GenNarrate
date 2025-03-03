from diffusers import StableDiffusionPipeline
import torch

# Load the image generation pipeline
def load_image_generator_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_attention_slicing()  # Reduce memory usage
    pipe.enable_sequential_cpu_offload()  # Offload model to CPU when not in use
    return pipe

# Generate images based on story
def generate_images_from_story(story, pipe, num_images=3, width=256, height=256):
    print("\nGenerating images for the story...")
    sentences = story
    prompts = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 10]
    image_prompts = prompts[:num_images]

    images = []
    for i, prompt in enumerate(image_prompts, 1):
        print(f"Generating image {i}/{num_images} for prompt: '{prompt}'")
        image = pipe(prompt, width=width, height=height).images[0]  # Adjust resolution
        images.append((prompt, image))

    return images

# Main function for image generation
def main_image_generation():
    # Load the saved story
    with open("generated_story.txt", "r") as file:
        story = file.read()

    print("\nStory Loaded:\n")
    print(story)
    print("\n-------------------\n")

    # Load the image generation model
    print("Loading image generator...")
    pipe = load_image_generator_model()

    # Prompt user for the number of images
    num_images = int(input("How many images do you want to generate? (default 3): ").strip() or 3)

    # Generate images
    images = generate_images_from_story(story, pipe, num_images)

    # Display the images
    print("\nDisplaying images...")
    for i, (prompt, image) in enumerate(images, 1):
        print(f"\nPrompt for Image {i}: {prompt}")
        display(image)

    print("\nImage generation completed.")


if __name__ == "__main__":
    main_image_generation()