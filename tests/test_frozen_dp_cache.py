from unittest.mock import patch
import torch
from maidata.model import build_model, build_draft, build_planner
from maidata.draft.windows import assemble_full_p0, select_windows
from maidata.decoder_training.p0_cache import save_bank, SCHEMA as P0_SCHEMA
from maidata.decoder_training.dp_cache import CachedP0Dataset, save_outputs
from maidata.training.data import collate_audio
from test_pretrain_runtime import config, item


def test_cached_p0_produces_exact_conditioned_z_and_tail_plan(config,tmp_path):
    source = item(0,config,True)
    online = collate_audio([source],config=config)
    encoder,draft,planner = build_model(config).eval(),build_draft(config).eval(),build_planner(config).eval()
    with torch.no_grad():
        local = encoder(online.songs,online.audio_windows)
        bank = assemble_full_p0(local.P0,online.audio_windows.target_start_beats,
            online.audio_windows.window_to_song,online.coverages,p0_rate=3,layout=online.layouts['p0'])
        expected_z = draft(bank,online.draft_inputs)
        expected_plan = planner(select_windows(expected_z,online.groups,online.starts,online.valid,
            target_beats=3,before_beats=0,after_beats=0,layout=online.layouts['planner']))
    row = save_bank(tmp_path/'banks/0.pt',bank,0,source['coverage'],variant='0',producer_sha256='p',p0_rate=3)
    row['path'] = 'banks/0.pt'
    keys = list(source['conditions'])
    windows = [dict(audio_variant='0',chart_id=w[1],window_id=w[0],start_beat=w[2]) for w in source['windows']]
    import json
    (tmp_path/'windows.jsonl').write_text('\n'.join(json.dumps(w) for w in windows))
    plan = dict(audio_groups=[dict(audio_variant='0',chart_ids=keys,encoder_starts_beats=[0],draft_end_beat=6)],
        charts=[dict(chart_id=k,chart_cache_path=k.replace(':','_')) for k in keys],policy=dict(split='train'),
        files=dict(windows='windows.jsonl'))
    c = dict(config,decoder_train_train_plan=str(tmp_path),decoder_train_chart_cache=str(tmp_path),
             decoder_train_source_root=str(tmp_path))
    manifest = dict(schema=P0_SCHEMA,producer=dict(sha256='p'),groups={'0':row})
    def chart(path):
        key = path.name.replace('_',':')
        condition = source['conditions'][key]
        return dict(source=dict(chart_id=key,audio_variant='0',song_id='0',difficulty=str(condition.difficulty_id),
            level_text=condition.level_text))
    with patch('maidata.decoder_training.dp_cache.load_chart',side_effect=chart):
        batch = CachedP0Dataset(c,tmp_path,manifest,[plan])[0].to('cpu')
    with torch.no_grad():
        z = draft(batch['bank'],batch['inputs'])
        predicted = planner(select_windows(z,batch['groups'],batch['starts'],batch['valid'],target_beats=3,
            before_beats=0,after_beats=0,layout=batch['layout']))
    assert torch.equal(z.values,expected_z.values)
    assert torch.equal(predicted.plan_values,expected_plan.plan_values)
    assert z.values.shape[:2] == (2,6)
    assert predicted.valid.all() and len(batch['window_ids']) == 4
    meta = save_outputs(tmp_path/'dp/0.pt',batch,z,predicted,'p')
    saved = torch.load(tmp_path/'dp/0.pt',weights_only=True)
    assert meta['windows']==4 and saved['chart_to_group']=={'0:5':0,'0:6':1}
    assert torch.equal(saved['z']['values'],z.values)
    assert torch.equal(saved['planner']['plan_values'],predicted.plan_values)
