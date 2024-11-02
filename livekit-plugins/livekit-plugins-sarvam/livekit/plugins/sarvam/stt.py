from __future__ import annotations

import dataclasses
import io
import os
import wave
from dataclasses import dataclass

import httpx
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.utils import AudioBuffer


@dataclass
class _STTOptions:
    language_code: str
    model: str
    with_timestamps: str


class STT(stt.STT):
    def __init__(
        self,
        *,
        language_code: str = "kn-IN",
        model: str = "saarika:v1",
        with_timestamps: str = "false",
        api_key: str | None = None,
        base_url: str = "https://api.sarvam.ai/speech-to-text",
    ):
        """
        Create a new instance of Sarvam STT.

        ``api_key`` must be set to your Sarvam API key, either using the argument or by setting the
        ``SARVAM_API_KEY`` environmental variable.
        """

        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )

        self._opts = _STTOptions(
            language_code=language_code,
            model=model,
            with_timestamps=with_timestamps,
        )

        self._api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self._api_key:
            raise ValueError("Sarvam API key is required")

        self._base_url = base_url
        self._client = httpx.AsyncClient(
            headers={"api-subscription-key": self._api_key},
            timeout=httpx.Timeout(connect=15.0, read=60.0, write=5.0, pool=5.0),
        )

    def _sanitize_options(self, *, language_code: str | None = None) -> _STTOptions:
        config = dataclasses.replace(self._opts)
        config.language_code = language_code or config.language_code
        return config

    async def _recognize_impl(
        self, buffer: AudioBuffer, *, language: str | None = None
    ) -> stt.SpeechEvent:
        try:
            config = self._sanitize_options(language_code=language)
            buffer = utils.merge_frames(buffer)
            io_buffer = io.BytesIO()
            with wave.open(io_buffer, "wb") as wav:
                wav.setnchannels(buffer.num_channels)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(buffer.sample_rate)
                wav.writeframes(buffer.data)
            io_buffer.seek(0)

            # Prepare the payload for Sarvam API
            files = {"file": ("audio.wav", io_buffer.read(), "audio/wav")}
            data = {
                "model": config.model,
                "language_code": config.language_code,
                "with_timestamps": config.with_timestamps,
            }
            print("STT -- request")
            print(data)

            # Call the Sarvam Speech-to-Text API
            response = await self._client.post(
                self._base_url,
                files=files,
                data=data,
            )
            # response.raise_for_status()

            result = response.json()
            print("STT -- response")
            print(result)
            transcript = result.get("transcript", "")

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(text=transcript, language=config.language_code)
                ],
            )

        except httpx.TimeoutException:
            raise APITimeoutError()
        except httpx.HTTPStatusError as e:
            raise APIStatusError(
                e.response.text,
                status_code=e.response.status_code,
                request_id=e.response.headers.get("X-Request-ID", ""),
                body=e.response.text,
            )
        except Exception as e:
            raise APIConnectionError() from e
