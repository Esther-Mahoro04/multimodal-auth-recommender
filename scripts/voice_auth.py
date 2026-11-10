"""Voiceprint verification (Member 3)."""
from config import VOICE_MODEL  # noqa: F401


def authenticate_voice(audio_path):
    """Take an audio file path. Return the member's name (str) or None if unknown."""
    # TODO (Member 3): load VOICE_MODEL, extract MFCC etc. from the audio, predict.
    # Return None when the voice is not an approved member (unauthorized attempt).
    raise NotImplementedError("voice_auth.authenticate_voice is not implemented yet")
