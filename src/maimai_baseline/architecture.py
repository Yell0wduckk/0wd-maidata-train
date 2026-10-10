"""Fixed native AR / chart decoder architecture, independent of assets."""
DECODER_LAYERS = 8
INITIAL_DECODER_LAYERS = 4  # Preserve the unchanged construction stream.
DECODER_LAYER_SEED_XOR = 0x44454336
PRIOR_SEED_XOR = 0x41525052
DENSE_DECODER_LAYER_PARAMETERS = 1053440
PRESERVED_DECODER_LAYERS = 6  # Fixed-seed prefix of the six-layer source.
NEW_DECODER_LAYER_SEED_XOR = 0x44454338
