import wave
import base64
from loguru import logger
import aiohttp
import asyncio

# Example audio parameters
sample_rate = 16000  # Hz
num_channels = 1  # Mono
sample_width = 2  # Bytes per sample (16-bit audio)

async def tts():
    text = "Hello, I am your friendly AI companion here to help you with any questions or tasks you might have."
    logger.info(f"Generating TTS for the first time: [{text}]")
                
    # url = "https://revapi.reverieinc.com/"
    # api_key = '84148cc0e57e75c7d1b1331bb99a2e94aa588d48'
    # speaker = "en_male"
    # format = "wav"
    # speed = 1.2
    # pitch = 1

    # payload = {
    #     "format": format,
    #     "speed": speed,
    #     "pitch": pitch,
    #     "segment": True,
    #     "cache": True,
    #     "text": text,
    #     "sample_rate": 16000,
    # }

    # headers = {
    #     "accept": "application/json, text/plain, */*",
    #     "content-type": "application/json",
    #     "rev-api-key": api_key,
    #     "rev-app-id": "rev.stt_tts",
    #     "rev-appname": "tts",
    #     "speaker": speaker,
    # }
    url = "https://waves-api.smallest.ai/api/v1/lightning/get_speech"

    payload = {

    "voice_id": "nisha",
    "text": "मैं जल जीवन मिशन से संगीता हूं, हम जल आपूर्ति पर आपके विचार और हमारी सेवाओं से आपकी संतुष्टि सुनना चाहेंगे ताकि हमें सुधार करने में मदद मिल सके, क्या आपके पास कुछ मिनट का समय है आगे बढ़ने के लिए?",
    "speed": 0.9,
    "sample_rate": 16000,
    "add_wav_header": True
    }

    headers = {
        # "accept": "application/json, text/plain, */*",
        "content-type": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI2NzU5MjNjZTlmZDgxMzQyM2E0ZTY5ZTQiLCJ0eXBlIjoiYXBpS2V5IiwiaWF0IjoxNzMzODk1MTE4LCJleHAiOjQ4ODk2NTUxMTh9.ABa821KIIFHlhJo6y3ziRLKxrkZj2I_S5EM-sHQZ9Y4",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload, headers=headers
        ) as r:
            if r.status != 200:
                error_text = await r.text()
                logger.error(
                    f"Error getting audio (status: {r.status}, error: {error_text})"
                )
                return f"Error getting audio (status: {r.status}, error: {error_text})"

            audio_data = await r.read()
            # print(audio_data)
            filename = "audio_tts.wav"
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(num_channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
            logger.info(f"Audio saved as {filename}")
            return "Audio generation successful"

if __name__ == '__main__':
    asyncio.run(tts())
