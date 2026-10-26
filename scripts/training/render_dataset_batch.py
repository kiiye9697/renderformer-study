"""Same locked data, official conversion in one process and one Blender worker."""
from pathlib import Path
import sys,json,time,hashlib,subprocess,contextlib
ROOT=Path(__file__).resolve().parents[2]
DATA=Path(sys.argv[1] if len(sys.argv)>1 else 'F:/RenderFormer-work/training-pilot/dataset-v1')
sys.path.insert(0,str(ROOT/'vendor/renderformer/scene_processor'))
from dacite import from_dict,Config
from scene_config import SceneConfig
from scene_mesh import generate_scene_mesh
from to_h5 import save_to_h5
m=json.loads((DATA/'manifest.json').read_text())
assert hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()==(DATA/'manifest.sha256').read_text().strip()
queue=[]
for frame in m['frames']:
    folder=DATA/Path(frame['config']).parent
    assert hashlib.sha256((DATA/frame['config']).read_bytes()).hexdigest()==frame['config_sha256']
    if (folder/'generation.json').exists() and (DATA/frame['reference']).exists():continue
    start=time.perf_counter()
    with (folder/'conversion.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        scene=from_dict(data_class=SceneConfig,data=json.loads((DATA/frame['config']).read_text()),config=Config(check_types=True,strict=True))
        generate_scene_mesh(scene,str(folder/'mesh/scene.obj'),str(folder))
        save_to_h5(scene,str(folder/'mesh/scene.obj'),str(DATA/frame['h5']))
    frame=dict(frame,conversion_seconds=time.perf_counter()-start);queue.append(frame)
    print('CONVERTED '+frame['id']+' '+str(frame['conversion_seconds']),flush=True)
(DATA/'render-queue.json').write_text(json.dumps(queue,indent=2))
with (DATA/'blender-batch.log').open('w') as log:
    subprocess.run(['C:/Program Files/Blender Foundation/Blender 4.5/blender.exe','-b','--factory-startup','--python',
       str(ROOT/'scripts/training/render_blender_batch.py'),'--','--dataset',str(DATA)],stdout=log,stderr=subprocess.STDOUT,check=True)
