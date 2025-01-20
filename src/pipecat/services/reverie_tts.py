import aiohttp
import os
from typing import AsyncGenerator
from pipecat.frames.frames import AudioRawFrame, ErrorFrame, Frame,TTSAudioRawFrame
from pipecat.services.ai_services import TTSService
from loguru import logger
import time
import wave
import base64

# Example audio parameters
sample_rate = 16000  # Hz
num_channels = 1  # Mono
sample_width = 2  # Bytes per sample (16-bit audio)
class ReverieTTSService(TTSService):

    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession,
        api_key: str = '84148cc0e57e75c7d1b1331bb99a2e94aa588d48',
        speaker: str = "en_male",
        format: str = "wav",
        speed: float = 1.2,
        pitch: float = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._api_key = api_key
        self._speaker = speaker
        self._aiohttp_session = aiohttp_session
        self._format = format
        self._speed = speed
        self._pitch = pitch

    def can_generate_metrics(self) -> bool:
        return True

    def generate_file_path(self, text: str) -> str:
        cache_dir = "cache/tts_cache"
        os.makedirs(cache_dir, exist_ok=True)
    
        # Add a unique identifier to the filename to avoid conflicts
        unique_id = str(hash(text))
        file_path = os.path.join(cache_dir, f"{unique_id}.{self._format}")
        
        return file_path

    def save_audio_data(self, text, audio_data):
        try:
            file_path = self.generate_file_path(f"{text}_{self._speaker}_{self._format}_{self._speed}_{self._pitch}")
            with open(file_path, "wb") as f:
                f.write(audio_data)
        except Exception as e:
            logger.error(f"Error saving audio data: {e}")

    def get_audio_data(self, text):
        try:
            file_path = self.generate_file_path(f"{text}_{self._speaker}_{self._format}_{self._speed}_{self._pitch}")
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > 0:
                    with open(file_path, "rb") as f:
                        audio_data = f.read()
                    logger.debug(f"Audio data fetched from cache: {file_path}")
                    return audio_data
            return None
        except Exception as e:
            logger.error(f"Error fetching audio data: {e}")
            return None

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        # audio_data = self.get_audio_data(text)
        audio_data = None
        
        if audio_data:
            logger.debug(f"Audio data found in cache for text: {text}")
            await self.start_ttfb_metrics()
            for i in range(0, len(audio_data), 16000):
                await self.stop_ttfb_metrics()
                chunk = audio_data[i:i+16000]
                frame = AudioRawFrame(chunk, 16000, 1)
                yield frame
        else:
            # log that generating tts for the first time
            logger.info(f"Generating TTS for the first timeeeee: [{text}]")
            
            url = "https://revapi.reverieinc.com/"

            payload = {
                "format": self._format,
                "speed": self._speed,
                "pitch": self._pitch,
                "segment": True,
                "cache": True,
                "text": text,
                "sample_rate": 16000,
            }

            headers = {
                "accept": "application/json, text/plain, */*",
                "content-type": "application/json",
                "rev-api-key": self._api_key,
                "rev-app-id": "rev.stt_tts",
                "rev-appname": "tts",
                "speaker": self._speaker,
            }

            await self.start_ttfb_metrics()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, headers=headers
                ) as r:
                    if r.status != 200:
                        error_text = await r.text()
                        logger.error(
                            f"{self} error getting audio (status: {r.status}, error: {error_text})"
                        )
                        yield ErrorFrame(
                            f"Error getting audio (status: {r.status}, error: {error_text})"
                        )
                        return

                    audio_data = await r.read()

                    print("Audio saved as output.wav")
                    await self.stop_ttfb_metrics()
                    # audio_data = base64.b64decode(audio_data)
                    # filename = f"audio_{int(time.time() * 1000)}.wav"
                    # with wave.open(filename, "wb") as wf:
                    #     wf.setnchannels(num_channels)
                    #     wf.setsampwidth(sample_width)
                    #     wf.setframerate(sample_rate)
                    #     wf.writeframes(audio_data)
                    # frame = TTSAudioRawFrame(audio_data, 16000, 1)
                    # await self.push_frame(frame)
                    # yield frame
                    for i in range(0, len(audio_data), 16000):
                        chunk = audio_data[i:i+16000]
                        # Create a unique filename using the current timestamp
                        filename = f"audio_{int(time.time() * 1000)}.wav"
                        with wave.open(filename, "wb") as wf:
                            wf.setnchannels(num_channels)
                            wf.setsampwidth(sample_width)
                            wf.setframerate(sample_rate)
                            wf.writeframes(chunk)
                        # frame = AudioRawFrame(chunk, 16000, 1)
                        # yield frame
                        # chunk = base64.b64decode(chunk)
                        frame = TTSAudioRawFrame(chunk, 16000, 1)
                        await self.push_frame(frame)
                    
                        # save the audio data in the cache 
                        # self.save_audio_data(text, chunk)

