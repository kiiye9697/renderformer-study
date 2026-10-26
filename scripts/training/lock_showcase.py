"""Predeclare camera-only qualitative showcases, before checkpoint selection."""
from pathlib import Path
import math,json,hashlib,copy,os
ROOT=Path(__file__).resolve().parents[2]
OUT=Path('F:/RenderFormer-work/training-pilot/showcase-v1')
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'manifest.json').exists():raise SystemExit('Already locked')
sources={'metal':ROOT/'experiments/teapot/baseline/config.json','sponza':Path('F:/RenderFormer-work/sponza-architecture-experiment/config.json')}
frames=[]
for family,source in sources.items():
    base=json.loads(source.read_text())
    for o in base['objects'].values():o['mesh_path']=str((source.parent/o['mesh_path']).resolve())
    if family=='metal':
        cameras=[{'position':[1.9*math.sin(math.radians(a)),-1.9*math.cos(math.radians(a)),.05],
          'look_at':[0,0,-.05],'up':[0,0,1],'fov':40} for a in [-36,-24,-12,0,12,24]]
    else:
        cameras=[{'position':pos,'look_at':target,'up':[0,0,1],'fov':62} for pos,target in [
          ([0,-.4,-.15],[0,.25,-.08]),([-.1,-.38,-.15],[0,.25,-.08]),([.1,-.38,-.15],[0,.25,-.08]),
          ([0,.4,-.15],[0,-.25,-.08]),([-.1,.38,-.15],[0,-.25,-.08]),([.1,.38,-.15],[0,-.25,-.08])]]
    for i,camera in enumerate(cameras):
        fid=f'{family}-showcase-{i:02d}';folder=OUT/'frames'/fid;folder.mkdir(parents=True)
        cfg=copy.deepcopy(base);cfg['cameras']=[camera]
        for o in cfg['objects'].values():o['mesh_path']=os.path.relpath(o['mesh_path'],folder).replace('\\','/')
        data=json.dumps(cfg,indent=2).encode();(folder/'config.json').write_bytes(data)
        frames.append({'id':fid,'family':family,'split':'qualitative_showcase','config':f'frames/{fid}/config.json',
          'h5':f'frames/{fid}/scene.h5','reference':f'frames/{fid}/reference.exr','config_sha256':hashlib.sha256(data).hexdigest(),'camera':camera})
manifest={'locked_before_training':True,'purpose':'Fixed-scene camera-only qualitative showcase, never used for checkpoint or hyperparameter selection; not independent generalization test',
 'resolution':256,'cycles_spp':2048,'cycles_seed':11,'frames':frames}
data=json.dumps(manifest,indent=2).encode();(OUT/'manifest.json').write_bytes(data)
(OUT/'manifest.sha256').write_text(hashlib.sha256(data).hexdigest())
print(hashlib.sha256(data).hexdigest())
