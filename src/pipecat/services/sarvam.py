import requests
import base64
import wave
from loguru import logger
import aiohttp
import asyncio
import json
# Example audio parameters
sample_rate = 16000  # Hz
num_channels = 1  # Mono
sample_width = 2  # Bytes per sample (16-bit audio)

url = "https://api.sarvam.ai/text-to-speech"

payload = {
    "target_language_code": "kn-IN",
    "speaker": "meera",
    "inputs": ["ಧನ್ಯವಾದಗಳು! Ride ID ಸರಿಯಾಗಿದೆ. ನೀವು ನಿಮ್ಮ ಬ್ಯಾಂಕ್ ಸ್ಟೇಟ್ಮೆಂಟ್ ಅನ್ನು ಪರಿಶೀಲಿಸಿದ್ದೀರಾ Payment ಅಪ್ಡೇಟ್ಗಳಿಗಾಗಿ? ಕೆಲವೊಮ್ಮೆ ಪಾವತಿಗಳು ತಡವಾಗಬಹುದು."],

    "speech_sample_rate": 16000,
    "enable_preprocessing": True
}
headers = {
    "api-subscription-key": "f6e16bff-d0da-48e9-8253-76fff0c6eb5e",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

# print(response.text)
response_dict = json.loads(response.text)

audio_data = base64.b64decode(response_dict["audios"][0])
filename = "audio_tts.wav"
with wave.open(filename, "wb") as wf:
    wf.setnchannels(num_channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(sample_rate)
    wf.writeframes(audio_data)
# logger.info(f"Audio saved as {filename}")
print("Audio generation successful")
