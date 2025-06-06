from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter
from diffusers import AudioLDM2Pipeline
import torch, re, os, scipy

# Change OUTPUT_DIR to a subfolder for audio files
OUTPUT_DIR = os.path.join("outputs", "audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_audio_from_story(story, voice_type, music_prompt):
    try:
        # Split story into sentences or chunks
        text_chunks = re.split(r'(?<=[.!?])\s+', story.strip())
        audio_segments = []

        def get_tts(text):
            # Choose TLD based on voice_type
            return gTTS(text=text, lang="en", tld={"male": "co.uk", "child": "com.au"}.get(voice_type, "com"))

        for i, chunk in enumerate(text_chunks):
            tts = get_tts(chunk)
            fname = os.path.join(OUTPUT_DIR, f"chunk_{i}.mp3")
            tts.save(fname)
            segment = AudioSegment.from_mp3(fname)
            audio_segments.append(segment)

        # Concatenate all audio chunks
        narration = sum(audio_segments)
        narration = normalize(low_pass_filter(narration, 3000))
        narration.export(os.path.join(OUTPUT_DIR, "narration.wav"), format="wav")

        # Load AudioLDM model on GPU
        pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-music", torch_dtype=torch.float16).to("cuda")
        duration_sec = len(narration) // 1000
        music = pipe(prompt=music_prompt, audio_length_in_s=duration_sec, num_inference_steps=200).audios
        scipy.io.wavfile.write(os.path.join(OUTPUT_DIR, "music.wav"), rate=16000, data=music[0])

        final_music = AudioSegment.from_wav(os.path.join(OUTPUT_DIR, "music.wav"))
        final_audio = narration.overlay(final_music - 10)
        final_path = os.path.join(OUTPUT_DIR, "final_story.wav")
        final_audio.export(final_path, format="wav")

        return final_path
    except Exception as e:
        raise RuntimeError(f"Audio generation failed: {str(e)}")
