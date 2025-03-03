




# Main function to interact with the user
def main():
    print("Welcome to the Story Generator!")

    while True:
        print("\nDo you want to provide a custom prompt? (yes/no):")
        use_custom_prompt = input("Choice: ").strip().lower()

        if use_custom_prompt == "yes":
            print("\nEnter your custom prompt:")
            prompt = input("Prompt: ").strip()
        else:
            print("\nEnter the genre of the story (e.g., Fantasy, Mystery, Adventure):")
            genre = input("Genre: ").strip()
            if genre.lower() == "exit":
                print("Goodbye!")
                break

            print("Enter the emotions you want in the story (e.g., Happy, Suspenseful):")
            emotions = input("Emotions: ").strip()

            print("Enter the characters for the story (e.g., knight, dragon):")
            characters = input("Characters: ").strip()

            # Combine inputs into a detailed prompt
            prompt = (f"Write a {genre} story that evokes {emotions} emotions. "
                      f"Include characters such as {characters}.")

        print("\nEnter the story length in tokens (e.g., 200):")
        try:
            max_length = int(input("Story Length: ").strip())
        except ValueError:
            print("Invalid input. Using default length of 200 tokens.")
            max_length = 200

        print("\nGenerating your story...\n")

        # Generate the story
        story = generate_story_with_completion(prompt, max_length)
        print("--- Your Story ---\n")
        print(story)
        print("\n-------------------\n")


        with open("generated_story.txt", "w") as file:
            file.write(story)

if __name__ == "__main__":
    main()
