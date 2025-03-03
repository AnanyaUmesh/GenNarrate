#pip install --upgrade git+https://github.com/huggingface/diffusers.git transformers accelerate

import scipy
import torch
from diffusers import AudioLDM2Pipeline

repo_id = "anhnct/audioldm2_gigaspeech"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# define the prompts
prompt = "A female reporter is speaking"
transcript = "wish you have a good day"

# set the seed for generator
generator = torch.Generator("cuda").manual_seed(0)

# run the generation
audio = pipe(
    prompt,
    transcription=transcript,
    num_inference_steps=200,
    audio_length_in_s=10.0,
    num_waveforms_per_prompt=2,
    generator=generator,
    max_new_tokens=512,          #Must set max_new_tokens equa to 512 for TTS
).audios

# save the best audio sample (index 0) as a .wav file
scipy.io.wavfile.write("tts.wav", rate=16000, data=audio[0])