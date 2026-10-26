"""Sequential, resumable official conversion and Cycles data generation."""
from pathlib import Path
import json,subprocess,time,hashlib,argparse
ROOT=Path(__file__).resolve().parents[2]
DATA=Path('F:/RenderFormer-work/training-pilot/dataset-v1')
PY='F:/RenderFormer-work/venv/Scripts/python.exe'
BLENDER='C:/Program Files/Blender Foundation/Blender 4.5/blender.exe'
p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=80);p.add_argument('--family');p.add_argument('--train-only',action='store_true');args=p.parse_args()
m=json.loads((DATA/'manifest.json').read_text());records=[]
assert hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()==(DATA/'manifest.sha256').read_text().strip()
for frame in m['frames']:
    if args.family and frame['family']!=args.family:continue
    if args.train_only and frame['split']!='train':continue
    if len(records)>=args.limit:break
    folder=DATA/Path(frame['config']).parent
    assert hashlib.sha256((DATA/frame['config']).read_bytes()).hexdigest()==frame['config_sha256']
    done=folder/'generation.json'
    if done.exists() and (DATA/frame['reference']).exists():
        records.append(json.loads(done.read_text()));continue
    start=time.perf_counter()
    with (folder/'conversion.log').open('w') as log:
        subprocess.run([PY,str(ROOT/'vendor/renderformer/scene_processor/convert_scene.py'),str(DATA/frame['config']),
            '--mesh_path',str(folder/'mesh/scene.obj'),'--output_h5_path',str(DATA/frame['h5'])],stdout=log,stderr=subprocess.STDOUT,check=True)
    converted=time.perf_counter()
    with (folder/'reference.log').open('w') as log:
        subprocess.run([BLENDER,'-b','--factory-startup','--python',str(ROOT/'scripts/blender_reference.py'),'--',
            '--config',str(DATA/frame['config']),'--mesh',str(folder/'mesh/scene.obj'),'--output',str(folder/'reference'),
            '--resolution',str(m['resolution']),'--spp',str(m['cycles_spp']),'--seed',str(m['cycles_seed'])],stdout=log,stderr=subprocess.STDOUT,check=True)
    assert (DATA/frame['reference']).exists(),frame['id']
    r={'id':frame['id'],'split':frame['split'],'conversion_seconds':converted-start,'total_seconds':time.perf_counter()-start,
       'reference_sha256':hashlib.sha256((DATA/frame['reference']).read_bytes()).hexdigest()}
    done.write_text(json.dumps(r,indent=2));records.append(r)
    (DATA/'generation-progress.json').write_text(json.dumps({'completed':len(records),'total':80,'records':records},indent=2))
    print(json.dumps(r),flush=True)
