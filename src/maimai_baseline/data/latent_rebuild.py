"""Bind newly encoded full-profile mu to its original frozen A and encoder60."""
from maimai_baseline.runtime_versions import operational
from maimai_baseline.training.chart_encoder_checkpoint import LEGACY_CHECKPOINT_SCHEMA as CHECKPOINT_SCHEMA


def validate_decoder_latent(cache, parent):
    """Keep A provenance intact while accepting explicitly rebuilt targets/stats."""
    binding = parent.get("baseline_manifest", {}).get("frozen_latent")
    source = cache.manifest.get("rebuilt_from")
    if source is None:
        from maimai_baseline.data.prior_decoder import validate_derived_latent
        return validate_derived_latent(cache, binding)
    if binding is None or operational(source) != operational(binding):
        raise ValueError("rebuilt latent origin differs from frozen A mu binding")
    for name in ("schema", "chart_style", "dimension", "beats_per_token"):
        if cache.identity.get(name) != source.get(name):
            raise ValueError("rebuilt latent representation changed: " + name)
    if (cache.identity["data_binding"].get("vocabulary_sha256")
            != source["data_binding"].get("vocabulary_sha256")):
        raise ValueError("rebuilt latent vocabulary differs from frozen A")
    expected = dict(schema=CHECKPOINT_SCHEMA, epoch=59, completed_epochs=60,
                    representation="deterministic-mu", beats_per_token=4, dimension=128)
    imported = cache.manifest.get("producer", {}).get("encoder_import", {})
    original = parent.get("frozen_encoder_import", {})
    for name, value in expected.items():
        if imported.get(name) != value or original.get(name) != value:
            raise ValueError("rebuilt latent encoder60 differs from frozen A: " + name)
    if cache.manifest.get("condition_support") != "encoded-full-profile-window-slots":
        raise ValueError("rebuilt latent must encode all full-profile condition slots")
    # Fresh normalization/files belong to the new B binding, never to A metadata.
    norm = cache.manifest["normalization"]
    if norm.get("includes_posterior_variance") is not False:
        raise ValueError("rebuilt latent normalization must use deterministic mu only")
