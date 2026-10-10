# Frozen Draft and Planner export

`python -m maidata.decoder_training.dp_cache --config CONFIG.json --p0-cache P0_ROOT --output NEW_OUTPUT`

Consumes the existing complete P0 manifest and banks from the same stage-I checkpoint. Keeps producer architecture, source guards, normalization and split binding. Only Draft D and Planner P run on CUDA bf16, in eval/no_grad; Encoder and Decoder stay on CPU and are not forwarded. Uses 12 workers by default. Does not rebuild audio, fields, Physical or P0.

Each audio variant has one bank containing full-song Z values/beats/valid for unique exact difficulty/level conditions, chart-to-condition mapping, and predicted plan12 for every unique chart window including the complete empty tail. Complex extra occurrences reuse these window identities. Z includes full Draft coverage so future consumers can select their own Decoder context. No GT planner labels enter prediction. Plan input is target-only Z. Both train and validation splits are exported without changing their membership.

Schema `maidata.frozen-draft-planner.v1`. Manifest records original producer, P0 manifest identity, normalization, plans, precision, source and generated-file hashes. Newly generated outputs are checked for finite values and saved atomically; first-bank tensor roundtrip is exact. Existing accepted payloads are only read as needed, without a separate recursive rehash.

Export does not start training or connect the cache to the training consumer. The current online training source must be adapted separately before a cache-backed run can start.
