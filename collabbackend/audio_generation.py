from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter
from transformers import pipeline
import torch, re, os, numpy as np, scipy.io.wavfile
from google.colab import files

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_audio_from_story(story, voice_type, music_prompt):
    try:
        # Split story into chunks for TTS
        text_chunks = re.split(r'(?<=[.!?])\s+', story.strip())
        audio_segments = []

        def get_tts(text):
            return gTTS(text=text, lang="en", tld={"male": "co.uk", "child": "com.au"}.get(voice_type, "com"))

        # Generate narration segments and combine
        for i, chunk in enumerate(text_chunks):
            tts = get_tts(chunk)
            fname = os.path.join(OUTPUT_DIR, f"chunk_{i}.mp3")
            tts.save(fname)
            segment = AudioSegment.from_mp3(fname)
            audio_segments.append(segment)

        narration = sum(audio_segments)
        narration = normalize(low_pass_filter(narration, 3000))
        narration.export(os.path.join(OUTPUT_DIR, "narration.wav"), format="wav")

        # Load MusicGen pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        musicgen = pipeline("text-to-audio", model="facebook/musicgen-large", device=0 if device=="cuda" else -1)

        # Calculate approximate max_length tokens for MusicGen based on narration length
        # MusicGen expects max_length in number of audio tokens, roughly 4 tokens = 1 second of audio
        duration_sec = len(narration) // 1000
        max_length = duration_sec * 4  # approx

        # Generate music audio as numpy array, sample rate 32000 (MusicGen default)
        music_output = musicgen(music_prompt, max_length=max_length)

        music_audio = music_output["array"]  # numpy float32 array
        music_sample_rate = music_output["sampling_rate"]  # usually 32000

        # Resample music to 16000Hz to match narration
        from scipy.signal import resample
        music_audio_resampled = resample(music_audio, int(len(music_audio) * 16000 / music_sample_rate))

        # Save music wav file
        music_wav_path = os.path.join(OUTPUT_DIR, "music.wav")
        scipy.io.wavfile.write(music_wav_path, 16000, (music_audio_resampled * 32767).astype(np.int16))

        # Load music and narration with pydub
        final_music = AudioSegment.from_wav(music_wav_path)
        # Lower volume of music to avoid overpowering narration
        final_music = final_music - 10

        # Overlay music with narration
        final_audio = narration.overlay(final_music)

        final_path = os.path.join(OUTPUT_DIR, "final_story.wav")
        final_audio.export(final_path, format="wav")

        # Download in Colab
        files.download(final_path)

        return final_path

    except Exception as e:
        raise RuntimeError(f"Audio generation failed: {str(e)}")
