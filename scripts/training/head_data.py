"""Shared cached-head I/O; training reads train/val only."""
from pilot_probe import *
from safetensors.torch import load_file,save_file
from renderformer.layers.dpt import DPTHead
import hashlib
DATA=Path('F:/RenderFormer-work/training-pilot/dataset-v1')
CACHE=Path('F:/RenderFormer-work/training-pilot/features-v1')
MODEL_SHA='792bfdc27b3d1adf4f4b5803a6333d05aed940fbc7e659003e0de556cd5156a0'
def verify_manifest():
    actual=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()
    assert actual==(DATA/'manifest.sha256').read_text().strip()
    return actual
def verify_cache_parity():
    manifest_sha=verify_manifest()
    parity=json.loads((CACHE/'parity.json').read_text())
    assert parity['max_abs']<1e-4
    meta=json.loads((CACHE/(parity['training_frame']+'.json')).read_text())
    assert meta['split']=='train' and meta['manifest_sha256']==manifest_sha and meta['source_model_sha256']==MODEL_SHA
    assert meta['cached_head_hdr_parity_max_abs']==parity['max_abs']
    assert meta['feature_file_sha256']==hashlib.sha256((CACHE/(parity['training_frame']+'.safetensors')).read_bytes()).hexdigest()
    index=json.loads((CACHE/'index.json').read_text())
    assert len(index['records'])==80 and not index['test_labels_read'] and not index['test_predictions_evaluated']
    assert all(r['manifest_sha256']==manifest_sha and r['source_model_sha256']==MODEL_SHA for r in index['records'])
    verified=dict(parity,manifest_sha256=manifest_sha,source_model_sha256=MODEL_SHA,
      feature_file_sha256=meta['feature_file_sha256'],dtype=meta['dtype'],shapes=meta['shapes'],identity_bound_via='same-frame cache metadata containing the parity result')
    (CACHE/'parity-verified.json').write_text(json.dumps(verified,indent=2))
    return verified

def make_head(checkpoint=None):
    verify_cache_parity()
    c=json.loads((CACHE/'model-config.json').read_text())
    h=DPTHead(in_channels=c['view_transformer_latent_dim'],features=c['dpt_features'],out_channels=c['dpt_out_channels'],out_dim=3).cuda()
    h.load_state_dict(load_file(str(checkpoint or CACHE/'initial-head.safetensors')));return h
def load_items(frames):
    manifest_sha=verify_manifest()
    integrity=json.loads((DATA/'integrity.json').read_text())
    assert integrity['manifest_sha256']==manifest_sha and integrity['status']=='passed'
    locked={r['id']:r for r in integrity['frames']}
    items=[]
    for f in frames:
        meta=json.loads((CACHE/(f['id']+'.json')).read_text())
        assert meta['manifest_sha256']==manifest_sha and meta['source_model_sha256']==MODEL_SHA
        assert meta['input_sha256']==hashlib.sha256((DATA/f['h5']).read_bytes()).hexdigest()
        assert meta['input_sha256']==locked[f['id']]['h5_sha256']
        assert meta['feature_file_sha256']==hashlib.sha256((CACHE/(f['id']+'.safetensors')).read_bytes()).hexdigest()
        assert meta['split']==f['split'] and meta['group_id']==f['group_id']
        generation=json.loads((DATA/Path(f['config']).parent/'generation.json').read_text())
        assert generation['reference_sha256']==hashlib.sha256((DATA/f['reference']).read_bytes()).hexdigest()
        tensors=load_file(str(CACHE/(f['id']+'.safetensors')))
        features=[tensors[f'feature_{i}'] for i in range(4)]
        raw=iio.imread(DATA/f['reference'])[...,:3].copy()
        assert raw.shape==(256,256,3) and np.isfinite(raw).all()
        target=torch.from_numpy(raw).permute(2,0,1)[None]
        items.append({'frame':f,'features':features,'target_log':torch.log10(1+target.clamp_min(0)),
          'label_stats':{'minimum':float(raw.min()),'negative_count':int((raw<0).sum()),'policy':'clamp negative radiance to zero before log10(1+x)'}})
    return items
def batch(items):
    features=[torch.cat([x['features'][i] for x in items]).cuda() for i in range(4)]
    target=torch.cat([x['target_log'] for x in items]).cuda();return features,target
def predict_log(head,features):
    with torch.autocast('cuda',dtype=torch.float16):
        return F.elu(head([[x] for x in features],32,32,patch_size=8),alpha=1e-3)
@torch.no_grad()
def evaluate_loss(head,items,microbatch):
    head.eval();values=[];families={}
    for start in range(0,len(items),microbatch):
        part=items[start:start+microbatch];features,target=batch(part)
        losses=(predict_log(head,features).float()-target).square().mean(dim=(1,2,3)).cpu().tolist()
        values.extend(losses)
        for item,value in zip(part,losses):families.setdefault(item['frame']['family'],[]).append(value)
    return {'mean':sum(values)/len(values),'families':{k:sum(v)/len(v) for k,v in families.items()}}
def save_head(head,path):save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(path))
