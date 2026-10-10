"""Build the currently implemented part of the model."""


def build_model(config):
    from .audio.encoder import AudioEncoder

    return AudioEncoder(**config)
