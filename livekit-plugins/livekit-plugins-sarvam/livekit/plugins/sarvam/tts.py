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
import os
from dataclasses import dataclass
from typing import List

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)

from .log import logger
from .models import SarvamLanguages, SarvamTTSModels

BASE_URL = "https://api.sarvam.ai/v1"


@dataclass
class TTSOptions:
    language: SarvamLanguages | str
    model: SarvamTTSModels
    sample_rate: int
    word_tokenizer: tokenize.WordTokenizer


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: SarvamTTSModels = "indic-tts-male",
        language: SarvamLanguages = "en",
        sample_rate: int = 22050,
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(
            ignore_punctuation=False
        ),
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Sarvam AI TTS.

        Args:
            model (SarvamTTSModels): The model to use for text-to-speech. Defaults to "indic-tts-male".
            language (SarvamLanguages): The language code. Defaults to "en".
            sample_rate (int): Audio sample rate in Hz. Defaults to 22050.
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text.
            api_key (str | None): Sarvam AI API key. Can be set via argument or SARVAM_API_KEY environment variable.
            http_session (aiohttp.ClientSession | None): Optional HTTP session for API requests.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

        api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if api_key is None:
            raise ValueError("Sarvam AI API key is required")

        self._api_key = api_key
        self._opts = TTSOptions(
            language=language,
            model=model,
            sample_rate=sample_rate,
            word_tokenizer=word_tokenizer,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def synthesize(self, text: str) -> "ChunkedStream":
        return ChunkedStream(self, text, self._opts, self._api_key, self._ensure_session())


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        tts: TTS,
        text: str,
        opts: TTSOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(tts, text)
        self._opts = opts
        self._api_key = api_key
        self._session = http_session

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        request_id = utils.shortuuid()

        try:
            async with self._session.post(
                url=f"{BASE_URL}/tts/generate",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json={
                    "text": self._input_text,
                    "model": self._opts.model,
                    "language": self._opts.language,
                },
            ) as res:
                response = await res.json()
                audio_data = response["audio"]  # base64 encoded audio data

                # Convert base64 audio to PCM frames
                frames = utils.codecs.decode_base64_audio(
                    audio_data,
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                )

                for frame in frames:
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=request_id,
                            frame=frame,
                        )
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