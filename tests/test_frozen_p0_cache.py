from copy import deepcopy
import pytest
import torch
from maidata.decoder_training.p0_cache import collate, save_bank, SCHEMA
from maidata.model import build_model
from maidata.draft.windows import assemble_full_p0
from test_pretrain_runtime import config, item


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_public_p0_cache_exact_roundtrip_and_padding(config,tmp_path,dtype):
    row=item(0,config)
    batch=collate([(row['variant'],row['grid'],row['coverage'])],config=config).to('cpu')
    encoder=build_model(config).eval()
    with torch.no_grad():
        local=encoder(batch.songs,batch.windows)
        bank=assemble_full_p0(local.P0,batch.windows.target_start_beats,batch.windows.window_to_song,
                             batch.coverages,p0_rate=config['fpn_rates'][0],layout=batch.p0_layout)
    bank.values=bank.values.to(dtype)
    target=tmp_path/'bank.pt'
    metadata=save_bank(target,bank,0,row['coverage'],variant='0',producer_sha256='producer',p0_rate=3)
    saved=torch.load(target,weights_only=True)
    assert saved['schema']==SCHEMA and saved['producer_sha256']=='producer'
    assert metadata['shape']==[9,18] and metadata['dtype']==str(dtype)
    assert not saved['valid'].all()
    assert all(torch.equal(saved[k],getattr(bank,k)[0]) for k in ('values','beats','valid'))


def test_export_rejects_wrong_coordinates_and_nonfinite(config,tmp_path):
    row=item(0,config)
    batch=collate([(row['variant'],row['grid'],row['coverage'])],config=config).to('cpu')
    with torch.no_grad():
        local=build_model(config).eval()(batch.songs,batch.windows)
        bank=assemble_full_p0(local.P0,batch.windows.target_start_beats,batch.windows.window_to_song,
                             batch.coverages,p0_rate=3,layout=batch.p0_layout)
    for key in ('beats','values'):
        bad=deepcopy(bank)
        getattr(bad,key).flatten()[0]=float('nan')
        with pytest.raises(ValueError,match='invalid'):
            save_bank(tmp_path/'bad.pt',bad,0,row['coverage'],variant='0',producer_sha256='p',p0_rate=3)
    bank.beats+=.25
    with pytest.raises(ValueError,match='coordinates'):
        save_bank(tmp_path/'bad.pt',bank,0,row['coverage'],variant='0',producer_sha256='p',p0_rate=3)
