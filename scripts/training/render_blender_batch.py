"""Persistent Blender worker; reset all datablocks before every independent frame."""
import bpy,sys,json,time,runpy,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DATA=Path(sys.argv[sys.argv.index('--dataset')+1] if '--dataset' in sys.argv else 'F:/RenderFormer-work/training-pilot/dataset-v1')
m=json.loads((DATA/'manifest.json').read_text())
queue=json.loads((DATA/'render-queue.json').read_text())
for frame in queue:
    folder=DATA/Path(frame['config']).parent
    start=time.perf_counter()
    bpy.ops.wm.read_factory_settings(use_empty=False)
    sys.argv=['blender','--','--config',str(DATA/frame['config']),'--mesh',str(folder/'mesh/scene.obj'),
      '--output',str(folder/'reference'),'--resolution',str(m['resolution']),
      '--spp',str(m['cycles_spp']),'--seed',str(m['cycles_seed'])]
    runpy.run_path(str(ROOT/'scripts/blender_reference.py'),run_name='__main__')
    camera=bpy.context.scene.camera
    (folder/'reference.camera.json').write_text(json.dumps({'matrix_world':[list(row) for row in camera.matrix_world],
      'fov_radians':float(camera.data.angle)},indent=2))
    assert (DATA/frame['reference']).exists()
    r={'id':frame['id'],'split':frame['split'],'conversion_seconds':frame['conversion_seconds'],
       'reference_worker_seconds':time.perf_counter()-start,
       'reference_sha256':hashlib.sha256((DATA/frame['reference']).read_bytes()).hexdigest(),
       'execution':'persistent worker, full factory reset per frame'}
    (folder/'generation.json').write_text(json.dumps(r,indent=2))
    count=sum((DATA/Path(f['config']).parent/'generation.json').exists() for f in m['frames'])
    (DATA/'generation-progress.json').write_text(json.dumps({'completed':count,'total':len(m['frames']),'last':r},indent=2))
    print('FRAME_COMPLETE '+json.dumps(r),flush=True)
