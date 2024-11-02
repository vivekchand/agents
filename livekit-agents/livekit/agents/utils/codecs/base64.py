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

import base64
import numpy as np


def decode_base64_audio(base64_audio: str, sample_rate: int, num_channels: int) -> list[np.ndarray]:
    """
    Decode base64 encoded audio data into PCM frames.
    
    Args:
        base64_audio: Base64 encoded audio data
        sample_rate: Audio sample rate in Hz
        num_channels: Number of audio channels
        
    Returns:
        List of numpy arrays containing PCM frames
    """
    # Decode base64 string to bytes
    audio_bytes = base64.b64decode(base64_audio)
    
    # Convert bytes to numpy array of 16-bit integers
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    
    # Reshape array if multi-channel
    if num_channels > 1:
        audio_array = audio_array.reshape(-1, num_channels)
    
    # Calculate frame size (20ms chunks)
    frame_size = int(sample_rate * 0.02)
    
    # Split into frames
    frames = []
    for i in range(0, len(audio_array), frame_size):
        frame = audio_array[i:i + frame_size]
        if len(frame) == frame_size:  # Only add complete frames
            frames.append(frame)
            
    return frames