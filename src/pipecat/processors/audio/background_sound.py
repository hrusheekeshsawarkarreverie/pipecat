#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

import numpy as np

from pipecat.audio.utils import resample_audio
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    CancelFrame,
    OutputAudioRawFrame,
    Frame,
    EndFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)

from loguru import logger

try:
    import soundfile as sf
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use background sound, you need to `pip install pipecat-ai[soundfile]`."
    )
    raise Exception(f"Missing module: {e}")


class BotBackgroundSound(FrameProcessor):
    def __init__(
        self,
        file_name: str,
        volume: float = 0.4,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._file_name = file_name
        self._volume = volume
        self._sample_rate = sample_rate

        self._sound = np.array([], dtype=np.int16)
        self._sound_pos = 0

        self._bot_speaking = False
        self._sleep_time = 0.02

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start()
            await self.push_frame(frame, direction)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSStartedFrame):
            self._bot_speaking = True
        elif isinstance(frame, TTSStoppedFrame):
            self._bot_speaking = False
        elif isinstance(frame, TTSAudioRawFrame):
            frame.audio = self._mix_with_sound(frame.audio)
            await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)

    async def _start(self):
        try:
            logger.debug(f"{self} loading background sound from {self._file_name}")
            sound, sample_rate = sf.read(self._file_name, dtype="int16")

            audio = sound.tobytes()
            if sample_rate != self._sample_rate:
                logger.debug(f"{self} resampling background sound to {self._sample_rate}")
                audio = resample_audio(audio, sample_rate, self._sample_rate)

            # Convert from np to bytes again.
            self._sound = np.frombuffer(audio, dtype=np.int16)

            self._audio_queue = asyncio.Queue()
            self._audio_task = self.get_event_loop().create_task(self._audio_task_handler())
        except Exception as ex:
            logger.error(f"{self} unable to open file {self._file_name}")

    async def _stop(self):
        self._audio_task.cancel()
        await self._audio_task

    def _mix_with_sound(self, audio: bytes):
        """Mixes raw audio frames with chunks of the same length from the sound
        file.

        """
        if audio:
            audio_np = np.frombuffer(audio, dtype=np.int16)
        else:
            num_samples = int(self._sleep_time * self._sample_rate)
            audio_np = np.zeros(num_samples, dtype=np.int16)

        chunk_size = len(audio_np)

        # Go back to the beginning if we don't have enough data.
        if self._sound_pos + chunk_size > len(self._sound):
            self._sound_pos = 0

        start_pos = self._sound_pos
        end_pos = self._sound_pos + chunk_size
        self._sound_pos = end_pos

        sound_np = self._sound[start_pos:end_pos]

        mixed_audio = np.clip(audio_np + sound_np * self._volume, -32768, 32767).astype(np.int16)

        return mixed_audio.astype(np.int16).tobytes()

    async def _audio_task_handler(self):
        while True:
            try:
                if not self._bot_speaking:
                    audio = self._mix_with_sound(b"")
                    frame = OutputAudioRawFrame(
                        audio=audio, sample_rate=self._sample_rate, num_channels=1
                    )
                    await self.push_frame(frame)
                await asyncio.sleep(self._sleep_time)
            except asyncio.CancelledError:
                break
