from typing import Literal

SarvamLanguages = Literal[
    "en",  # English
    "hi",  # Hindi
    "ta",  # Tamil
    "te",  # Telugu
    "kn",  # Kannada
    "ml",  # Malayalam
    "mr",  # Marathi
    "gu",  # Gujarati
    "pa",  # Punjabi
    "bn",  # Bengali
    "or",  # Odia
]

SarvamSTTModels = Literal[
    "whisper-large-v3",
]

SarvamTTSModels = Literal[
    "indic-tts-male",
    "indic-tts-female",
]