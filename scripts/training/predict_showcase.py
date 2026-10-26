"""Fixed predeclared camera-only views, only after final held-out test is complete."""
from pilot_probe import *
from safetensors.torch import load_file
import hashlib,shutil,h5py,copy
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate import display_transform,measure,plt
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();RUN=Path(a.run)
assert (RUN/'test-summary.json').exists(),'Qualitative showcase follows final held-out evaluation'
result=json.loads((RUN/'result.json').read_text());best_sha=hashlib.sha256((RUN/'best.safetensors').read_bytes()).hexdigest()
assert best_sha==result['checkpoint_sha256']['best']
DATA=Path('F:/RenderFormer-work/training-pilot/showcase-v1');m=json.loads((DATA/'manifest.json').read_text())
assert hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()==(DATA/'manifest.sha256').read_text().strip()
evidence={};family_hashes={}
for f in m['frames']:
    folder=DATA/Path(f['config']).parent;cfg=json.loads((folder/'config.json').read_text())
    assert hashlib.sha256((folder/'config.json').read_bytes()).hexdigest()==f['config_sha256']
    assert cfg['cameras'][0]==f['camera']
    objects=copy.deepcopy(cfg['objects'])
    for obj in objects.values():obj['mesh_sha256']=hashlib.sha256((folder/obj.pop('mesh_path')).read_bytes()).hexdigest()
    fixed_sha=hashlib.sha256(json.dumps(objects,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    family_hashes.setdefault(f['family'],set()).add(fixed_sha)
    geo=np.load(folder/'reference.geometry.npz');camera=json.loads((folder/'reference.camera.json').read_text())
    with h5py.File(DATA/f['h5'],'r') as h:
        vertex=float(np.max(np.abs(h['triangles'][:]-geo['triangles'])))
        normal=float(np.max(np.abs(h['vn'][:]-geo['vn'])))
        cam_error=float(np.max(np.abs(h['c2w'][0]-np.asarray(camera['matrix_world']))))
        fov_error=abs(float(h['fov'][0])-camera['fov_radians']*180/np.pi)
    assert vertex<1e-6 and normal<.001 and cam_error<1e-5 and fov_error<1e-4,(f['id'],vertex,normal,cam_error,fov_error)
    gen=json.loads((folder/'generation.json').read_text())
    assert hashlib.sha256((folder/'reference.exr').read_bytes()).hexdigest()==gen['reference_sha256']
    evidence[f['id']]={'config_sha256':f['config_sha256'],'geometry_material_light_sha256':fixed_sha,
      'alignment':{'vertex_max_abs':vertex,'normal_max_abs':normal,'camera_matrix_max_abs':cam_error,'fov_degrees_abs':fov_error}}
assert all(len(values)==1 for values in family_hashes.values())
pipe=RenderFormerRenderingPipeline.from_pretrained('F:/RenderFormer-work/models/renderformer-v1-base');pipe.to(torch.device('cuda'));pipe.model.eval()
for label in ['pretrained','finetuned']:
    if label=='finetuned':pipe.model.view_transformer.out_dpt.load_state_dict(load_file(str(RUN/'best.safetensors')))
    for f in m['frames']:
        folder=DATA/Path(f['config']).parent
        assert hashlib.sha256((folder/'config.json').read_bytes()).hexdigest()==f['config_sha256']
        inputs={k:v.unsqueeze(0).cuda() for k,v in load_single_h5_data(str(DATA/f['h5'])).items()};inputs['fov']=inputs['fov'].unsqueeze(-1)
        with torch.no_grad():pred=pipe(**inputs,resolution=256,torch_dtype=torch.float16)[0,0].float().cpu().numpy()
        assert pred.shape==(256,256,3) and np.isfinite(pred).all()
        iio.imwrite(folder/(label+'.exr'),pred);del inputs
OUT=RUN/'publication';ASSETS=OUT/'assets/showcase';ASSETS.mkdir(parents=True,exist_ok=True);cases=[]
for f in m['frames']:
    folder=DATA/Path(f['config']).parent;ref=iio.imread(folder/'reference.exr')[...,:3];row={k:f[k] for k in ['id','family','camera']};row['metrics']={};row.update(evidence[f['id']])
    for key in ['reference','pretrained','finetuned']:
        raw=ref if key=='reference' else iio.imread(folder/(key+'.exr'))
        filename=f['id']+'-'+key+'.png';iio.imwrite(ASSETS/filename,np.round(display_transform(raw)*255).astype(np.uint8));row[key]='assets/showcase/'+filename
        if key!='reference':
            row['metrics'][key]=measure(ref,raw)
            error=np.mean(np.abs(display_transform(ref)-display_transform(raw)),axis=-1)
            filename=f['id']+'-'+key+'-error.png';plt.imsave(ASSETS/filename,error,cmap='magma',vmin=0,vmax=.25);row[key+'_error']='assets/showcase/'+filename
            row['metrics'][key].update(negative_prediction_fraction=float(np.mean(raw<0)),error_saturated_pixel_fraction=float(np.mean(error>=.25)))
    row.update(error_scale=[0,.25],error_note='Display RGB MAE; values >=0.25 saturate at maximum magma color')
    row['shape']=list(ref.shape);cases.append(row)
(OUT/'showcase-summary.json').write_text(json.dumps({'status':'complete','purpose':m['purpose'],
 'manifest_sha256':(DATA/'manifest.sha256').read_text().strip(),'cases':cases,'checkpoint_sha256':best_sha,
 'family_fixed_condition_sha256':{family:next(iter(values)) for family,values in family_hashes.items()},
 'fixed_condition_hash_definition':'SHA256 of canonical object configuration including transforms/materials/lights, replacing source mesh paths with actual mesh file SHA256; excludes camera',
 'display_transform':'EV0; clamp negative for display; channel-wise Reinhard; sRGB; displayMAE fixed[0,.25]',
 'labels':'A=Cycles reference; B=pretrained; C=finetuned','camera_sequence':'Discrete actual renders; no image interpolation or implied continuous neural rendering'},indent=2))
print(OUT/'showcase-summary.json')
