"""CLI imports Torch only when executing, after help/config processing."""
import argparse
import json
from pathlib import Path
from maidata.entry import CONFIG


def main(argv=None, *, adaptation=False):
    parser = argparse.ArgumentParser(description=('Adapt M to frozen P0/Z caches; only Decoder updates.' if adaptation else
        'Stage II: train a fresh audio Encoder and Decoder with fixed Draft/Plan.'))
    parser.add_argument('--config',type=Path,help='JSON overrides for entry.CONFIG')
    parser.add_argument('--plan-source',choices=('predicted','gt'),help='Override Decoder plan conditioning source')
    parser.add_argument('--gt-plan-cache',type=Path,help='Offline GT plan12 cache, required for gt mode')
    parser.add_argument('--output',type=Path,help='Override the independent training output directory')
    parser.add_argument('--init-checkpoint',type=Path,help='Existing M checkpoint for weight-only cached adaptation')
    parser.add_argument('--p0-cache',type=Path,help='Frozen stage-II Encoder public P0 cache including Encoder snapshot')
    parser.add_argument('--local-p0-cache',type=Path,help='Frozen exact local-window P0 for Decoder audio')
    parser.add_argument('--dp-cache',type=Path,help='Frozen split Drafter Z/plan cache')
    parser.add_argument('--resume',type=Path,help='v4 complete-epoch checkpoint of the same training mode')
    parser.add_argument('--show-config',action='store_true')
    parser.add_argument('--cuda-check',action='store_true',help='one actual configured update; no training checkpoint')
    parser.add_argument('--check-step',type=int,default=0,help='zero-based first-epoch logical batch for CUDA check')
    args = parser.parse_args(argv)
    config = dict(CONFIG)
    if args.config:
        values = json.loads(args.config.read_text(encoding='utf-8-sig'))
        if set(values)-set(config):
            parser.error('unknown configuration keys: '+str(sorted(set(values)-set(config))))
        config.update(values)
    for name, key in (('plan_source','decoder_train_plan_source'), ('gt_plan_cache','decoder_train_gt_plan_cache'),
                      ('output','decoder_train_output'), ('init_checkpoint','decoder_train_init_checkpoint'),
                      ('p0_cache','decoder_train_p0_cache'), ('dp_cache','decoder_train_dp_cache'),
                      ('local_p0_cache','decoder_train_local_p0_cache')):
        value = getattr(args,name)
        if value is not None:
            config[key] = str(value)
    if adaptation:
        config.update(decoder_train_audio_mode='cached', decoder_train_audio_scope='window',
                      decoder_z_cross_attention=True)
    if args.resume and args.init_checkpoint:
        parser.error('--resume and --init-checkpoint select different operations')
    if args.show_config:
        print(json.dumps(config,ensure_ascii=False,indent=2))
        return 0
    from .runner import run
    run(config,resume=args.resume,cuda_check=args.cuda_check,check_step=args.check_step)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
