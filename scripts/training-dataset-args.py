"""Resolve portable dataset.json paths to a JSON argv array for either launcher."""
import argparse
import json
from pathlib import Path
from maimai_baseline.data.training_dataset import read, inside, sha
from maimai_dataset.window import WINDOW


def arguments(manifest, recipe, *, validate=True):
    manifest=Path(manifest).resolve();root=manifest.parent;data=read(manifest)
    if data.get('schema')!='maimai.training_dataset.v1' or data.get('window')!=WINDOW or not data.get('ready'):
        raise ValueError('dataset is not verified for the current window')
    report=inside(root,data['verification'])
    if validate and (sha(report)!=data['verification_sha256'] or not read(report).get('passed')):
        raise ValueError('dataset verification identity mismatch')
    paths={**data['inputs'],'epoch_plan_cache':data['epoch_plans'][str(recipe)]}
    result=[]
    for name,value in paths.items():
        path=inside(root,value)
        if validate and not path.exists():raise ValueError(f'missing runtime path: {path}')
        result+=['--'+name.replace('_','-'),str(path)]
    result += ['--audio-policy','normalized-p50']
    if data.get('validation', {}).get('enabled') is False:
        result += ['--validation-policy', 'disabled']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-manifest',type=Path,required=True)
    p.add_argument('--recipe',choices=['3072','4096'],required=True)
    args=p.parse_args()
    print(json.dumps(arguments(args.dataset_manifest,args.recipe),ensure_ascii=False))


if __name__=='__main__':main()
