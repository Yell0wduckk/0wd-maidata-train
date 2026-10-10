"""Independent whole-song audio architecture and cache contract."""
CONTRACT = dict(schema='maimai.song_audio256_p2_fusion.v1', rate=1, width=256,
    patch=[24,128], tokenizer='ordered-mel-plus-valid-linear3096-gelu-ln',
    layers=2, heads=8, ffn=1024, dropout=.1, coordinate='absolute-beat-center',
    query_chunk=128, checkpoint=False, fusion_geometry='absolute-half-beat-origin-minus4;offline-linear-clamped-valid-renormalized', conditioning='absolute-linear-to-M2-projection-residual;P2-to-P0')

# Immutable metadata of the existing raw Mel pack; not its consumer architecture.
INPUT_CONTRACT = dict(schema='maimai.song_audio_p2_fusion.v1', rate=1, width=192,
    patch=[24,128], tokenizer='ordered-mel-plus-valid-linear3096-gelu-ln',
    layers=2, heads=6, ffn=768, dropout=.1, coordinate='absolute-beat-center',
    query_chunk=128, checkpoint=False, fusion_geometry='absolute-half-beat-origin-minus4;offline-linear-clamped-valid-renormalized', conditioning='absolute-linear-to-M2-projection-residual;P2-to-P0')
