from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter
from transformers import pipeline
import torch, re, os, numpy as np, scipy.io.wavfile
from google.colab import files
import glob
from scipy.signal import resample

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_audio_from_story(story, voice_type, music_prompt):
    try:
        # Step 1: Split story into chunks
        text_chunks = re.split(r'(?<=[.!?])\s+', story.strip())
        audio_segments = []

        def get_tts(text):
            tld_map = {"male": "co.uk", "child": "com.au"}
            return gTTS(text=text, lang="en", tld=tld_map.get(voice_type, "com"))

        # Step 2: Generate audio segments for each chunk
        for i, chunk in enumerate(text_chunks):
            tts = get_tts(chunk)
            fname = os.path.join(OUTPUT_DIR, f"chunk_{i}.mp3")
            tts.save(fname)
            segment = AudioSegment.from_mp3(fname)
            audio_segments.append(segment)

        # Combine narration and apply audio effects
        narration = sum(audio_segments)
        narration = normalize(narration)  # Optional: low_pass_filter(narration, 3000)
        narration.export(os.path.join(OUTPUT_DIR, "narration.wav"), format="wav")

        # Step 3: Generate background music with MusicGen
        device = "cuda" if torch.cuda.is_available() else "cpu"
        musicgen = pipeline("text-to-audio", model="facebook/musicgen-large", device=0 if device == "cuda" else -1)

        duration_sec = len(narration) // 1000
        music_output = musicgen(music_prompt, max_duration_in_s=duration_sec)
        music_audio = music_output[0]["audio"]
        music_sample_rate = 32000  # MusicGen default

        # Step 4: Resample music to 16kHz
        if len(music_audio.shape) > 1:
            music_audio = music_audio.mean(axis=0)  # Convert to mono

        music_audio_resampled = resample(music_audio, int(len(music_audio) * 16000 / music_sample_rate))
        music_wav_path = os.path.join(OUTPUT_DIR, "music.wav")
        scipy.io.wavfile.write(music_wav_path, 16000, (music_audio_resampled * 32767).astype(np.int16))

        # Step 5: Mix narration with background music
        final_music = AudioSegment.from_wav(music_wav_path)
        final_music = final_music - 10  # Lower music volume

        final_audio = narration.overlay(final_music)
        final_path = os.path.join(OUTPUT_DIR, "final_story.wav")
        final_audio.export(final_path, format="wav")

        # Step 6: Clean up temporary TTS files
        for f in glob.glob(os.path.join(OUTPUT_DIR, "chunk_*.mp3")):
            os.remove(f)

        # Step 7: Download in Colab
        files.download(final_path)
        return final_path

    except Exception as e:
        raise RuntimeError(f"Audio generation failed: {str(e)}")
