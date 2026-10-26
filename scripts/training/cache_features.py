"""Precompute frozen DPT inputs. Test labels/predictions are not evaluated here."""
from pilot_probe import *
from safetensors.torch import save_file,load_file
import hashlib
DATA=Path('F:/RenderFormer-work/training-pilot/dataset-v1')
CACHE=Path('F:/RenderFormer-work/training-pilot/features-v1');CACHE.mkdir(parents=True,exist_ok=True)
m=json.loads((DATA/'manifest.json').read_text())
manifest_sha=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()
assert manifest_sha==(DATA/'manifest.sha256').read_text().strip()
model_sha=hashlib.sha256(Path('F:/RenderFormer-work/models/renderformer-v1-base/model.safetensors').read_bytes()).hexdigest()
assert model_sha=='792bfdc27b3d1adf4f4b5803a6333d05aed940fbc7e659003e0de556cd5156a0'
pipe=RenderFormerRenderingPipeline.from_pretrained('F:/RenderFormer-work/models/renderformer-v1-base');pipe.to(torch.device('cuda'))
pipe.model.eval()
for par in pipe.model.parameters():par.requires_grad_(False)
head=pipe.model.view_transformer.out_dpt
initial={k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()}
if not (CACHE/'initial-head.safetensors').exists():save_file(initial,str(CACHE/'initial-head.safetensors'))
else:
    saved=load_file(str(CACHE/'initial-head.safetensors'))
    assert saved.keys()==initial.keys() and all(torch.equal(saved[k],v) for k,v in initial.items())
cfg=json.loads(Path('F:/RenderFormer-work/models/renderformer-v1-base/config.json').read_text())
(CACHE/'model-config.json').write_text(json.dumps(cfg,indent=2))
class Captured(Exception):pass
captured={};allow_output=False
def capture(module,args,kwargs):
    assert all(isinstance(x,(list,tuple)) and torch.is_tensor(x[0]) for x in args[0])
    # Official DPTHead.forward consumes only x[0] from each layer container.
    captured['container_lengths']=[len(x) for x in args[0]]
    captured['features']=[x[0].detach().cpu().contiguous() for x in args[0]]
    captured['patch_h']=args[1];captured['patch_w']=args[2];captured['patch_size']=kwargs['patch_size']
    if not allow_output:raise Captured()
hook=head.register_forward_pre_hook(capture,with_kwargs=True)
records=[];parity_done=False
for f in m['frames']:
    h5=DATA/f['h5'];assert h5.exists(),h5
    dest=CACHE/(f['id']+'.safetensors');meta=CACHE/(f['id']+'.json')
    if dest.exists() and meta.exists():
        r=json.loads(meta.read_text());assert r['input_sha256']==hashlib.sha256(h5.read_bytes()).hexdigest()
        assert r['manifest_sha256']==manifest_sha and r['source_model_sha256']==model_sha
        assert r['split']==f['split'] and r['group_id']==f['group_id']
        assert r['feature_file_sha256']==hashlib.sha256(dest.read_bytes()).hexdigest()
        records.append(r);continue
    data={k:v.unsqueeze(0).cuda() for k,v in load_single_h5_data(str(h5)).items()};data['fov']=data['fov'].unsqueeze(-1)
    captured.clear();allow_output=(not parity_done and f['split']=='train')
    start=time.perf_counter()
    try:
        with torch.no_grad():official=pipe(**data,resolution=256,torch_dtype=torch.float16)
    except Captured:pass
    assert len(captured['features'])==4
    features=captured['features']
    save_file({f'feature_{i}':x for i,x in enumerate(features)},str(dest))
    r={'id':f['id'],'group_id':f['group_id'],'family':f['family'],'split':f['split'],
       'patch_h':captured['patch_h'],'patch_w':captured['patch_w'],'patch_size':captured['patch_size'],
       'shapes':[list(x.shape) for x in features],'dtype':[str(x.dtype) for x in features],
       'source_container_lengths':captured['container_lengths'],'cache_layout':'Each feature_i is layer_container[0], the only element consumed by official DPTHead; reconstruct as [[feature_i], ...]',
       'input_sha256':hashlib.sha256(h5.read_bytes()).hexdigest(),'manifest_sha256':(DATA/'manifest.sha256').read_text().strip(),
       'source_model_sha256':'792bfdc27b3d1adf4f4b5803a6333d05aed940fbc7e659003e0de556cd5156a0',
       'seconds':time.perf_counter()-start,'test_prediction_computed':False}
    r['feature_file_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
    if allow_output:
        hook.remove()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            logpred=F.elu(head([[x.cuda()] for x in features],captured['patch_h'],captured['patch_w'],patch_size=captured['patch_size']),alpha=1e-3)
        # Match official rendering_pipeline: inverse transform outside autocast.
        hdr=(torch.pow(10.,logpred)-1).permute(0,2,3,1)[:,None]
        parity=float((hdr-official).abs().max());assert parity<1e-4,parity
        r['cached_head_hdr_parity_max_abs']=parity;parity_done=True
        (CACHE/'parity.json').write_text(json.dumps({'training_frame':f['id'],'max_abs':parity},indent=2))
        hook=head.register_forward_pre_hook(capture,with_kwargs=True)
    meta.write_text(json.dumps(r,indent=2));records.append(r)
    (CACHE/'progress.json').write_text(json.dumps({'completed':len(records),'total':80},indent=2))
    print(json.dumps({'id':f['id'],'cached':len(records),'seconds':r['seconds']}),flush=True)
    del data
hook.remove()
(CACHE/'index.json').write_text(json.dumps({'records':records,'test_labels_read':False,'test_predictions_evaluated':False},indent=2))
