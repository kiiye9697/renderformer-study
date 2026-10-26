"""Byte/geometry integrity only for test; no test reference pixels or metrics read."""
from pathlib import Path
import json,hashlib,h5py,numpy as np
DATA=Path('F:/RenderFormer-work/training-pilot/dataset-v1')
m=json.loads((DATA/'manifest.json').read_text());sha=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()
assert sha==(DATA/'manifest.sha256').read_text().strip()
records=[]
for f in m['frames']:
    folder=DATA/Path(f['config']).parent;gen=json.loads((folder/'generation.json').read_text())
    assert hashlib.sha256((DATA/f['config']).read_bytes()).hexdigest()==f['config_sha256']
    assert hashlib.sha256((DATA/f['reference']).read_bytes()).hexdigest()==gen['reference_sha256']
    geometry=np.load(folder/'reference.geometry.npz')
    with h5py.File(DATA/f['h5'],'r') as h:
        v=float(np.max(np.abs(h['triangles'][:]-geometry['triangles'])))
        n=float(np.max(np.abs(h['vn'][:]-geometry['vn'])))
        assert v<1e-6 and n<.001,(f['id'],v,n)
        triangles=len(h['triangles'])
    records.append({'id':f['id'],'split':f['split'],'triangles':triangles,'vertex_max_abs':v,'normal_max_abs':n,
      'h5_sha256':hashlib.sha256((DATA/f['h5']).read_bytes()).hexdigest(),'reference_sha256':gen['reference_sha256']})
assert len(records)==80
out={'status':'passed','manifest_sha256':sha,'frames':records,'test_pixels_read':False,'test_metrics_computed':False}
(DATA/'integrity.json').write_text(json.dumps(out,indent=2));print('PASS 80 frame hashes and Blender/H5 geometry')
