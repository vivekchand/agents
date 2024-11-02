# Copyright 2024 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import dataclasses
import io
import os
import wave
from dataclasses import dataclass
from typing import List

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.utils import AudioBuffer, merge_frames

from .log import logger
from .models import SarvamLanguages, SarvamSTTModels

BASE_URL = "https://api.sarvam.ai"

# This is the magic number during testing that we use to determine if a frame is loud enough
# to possibly contain speech. It's very conservative.
MAGIC_NUMBER_THRESHOLD = 0.004**2


class _AudioEnergyFilter:
    def __init__(self, *, min_silence: float = 1.5):
        self._cooldown_seconds = min_silence
        self._cooldown = min_silence
        self._speaking = False

    def update(self, frame: rtc.AudioFrame) -> bool:
        arr = np.frombuffer(frame.data, dtype=np.int16)
        float_arr = arr.astype(np.float32) / 32768.0
        rms = np.mean(np.square(float_arr))

        if rms > MAGIC_NUMBER_THRESHOLD:
            self._cooldown = self._cooldown_seconds
            self._speaking = True
        else:
            self._cooldown -= frame.duration
            if self._cooldown <= 0:
                self._speaking = False

        return self._speaking


@dataclass
class STTOptions:
    language: SarvamLanguages | str | None
    model: SarvamSTTModels
    sample_rate: int
    num_channels: int


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: SarvamSTTModels = "whisper-large-v3",
        language: SarvamLanguages = "en",
        sample_rate: int = 16000,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Sarvam AI STT.

        Args:
            model (SarvamSTTModels): The model to use for speech recognition. Defaults to "whisper-large-v3".
            language (SarvamLanguages): The language code. Defaults to "en".
            sample_rate (int): Audio sample rate in Hz. Defaults to 16000.
            api_key (str | None): Sarvam AI API key. Can be set via argument or SARVAM_API_KEY environment variable.
            http_session (aiohttp.ClientSession | None): Optional HTTP session for API requests.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=False,  # Sarvam AI doesn't support interim results
            )
        )

        api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if api_key is None:
            raise ValueError("Sarvam AI API key is required")

        self._api_key = api_key
        self._opts = STTOptions(
            language=language,
            model=model,
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self, buffer: AudioBuffer, *, language: SarvamLanguages | str | None = None
    ) -> stt.SpeechEvent:
        config = dataclasses.replace(self._opts)
        if language:
            config.language = language

        buffer = merge_frames(buffer)
        io_buffer = io.BytesIO()
        with wave.open(io_buffer, "wb") as wav:
            wav.setnchannels(buffer.num_channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(buffer.sample_rate)
            wav.writeframes(buffer.data)

        data = io_buffer.getvalue()

        try:
            async with self._ensure_session().post(
                url=f"{BASE_URL}/speech/recognition",
                data=data,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                    "Content-Type": "audio/wav",
                },
                params={
                    "model": config.model,
                    "language": config.language,
                },
            ) as res:
                response = await res.json()
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            language=config.language,
                            text=response["text"],
                            confidence=1.0,  # Sarvam AI doesn't provide confidence scores
                        )
                    ],
                )

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e

    def stream(
        self, *, language: SarvamLanguages | str | None = None
    ) -> "SpeechStream":
        config = dataclasses.replace(self._opts)
        if language:
            config.language = language
        return SpeechStream(self, config, self._api_key, self._ensure_session())


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        stt: STT,
        opts: STTOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._audio_energy_filter = _AudioEnergyFilter()
        self._buffer = AudioBuffer(
            sample_rate=opts.sample_rate,
            num_channels=opts.num_channels,
        )
        self._speaking = False

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        async for frame in self._input_ch:
            if isinstance(frame, self._FlushSentinel):
                if len(self._buffer) > 0:
                    event = await self._stt._recognize_impl(
                        self._buffer, language=self._opts.language
                    )
                    self._event_ch.send_nowait(event)
                    self._buffer.clear()
                continue

            is_speaking = self._audio_energy_filter.update(frame)
            if is_speaking:
                if not self._speaking:
                    self._speaking = True
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    )
                self._buffer.append(frame)
            elif self._speaking and len(self._buffer) > 0:
                self._speaking = False
                event = await self._stt._recognize_impl(
                    self._buffer, language=self._opts.language
                )
                self._event_ch.send_nowait(event)
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                )
                self._buffer.clear()