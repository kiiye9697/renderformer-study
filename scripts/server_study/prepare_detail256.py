import os
import json,copy,subprocess
from pathlib import Path
r=Path(os.environ['RENDERFORMER_WORKSPACE']);d=r/'datasets/detail256';d.mkdir(exist_ok=True)
m=json.loads((r/'datasets/training/manifest.json').read_text());m['resolution']=256;m['spp']=2048
for a in m['records']:
 c=json.loads(Path(a['config']).read_text())
 # Hold out a fresh material configuration; previous .7 tests remain development evidence.
 if a['split']=='test':
  for k,o in c['objects'].items():
   if k=='bunny' or (a['family']=='sponza' and k!='light_0'):o['material']['roughness']=.65
  a['roughness']=.65
 for o in c['objects'].values():o['mesh_path']=str((Path(a['config']).parent/o['mesh_path']).resolve())
 a['config']=str(d/(a['name']+'.json'));Path(a['config']).write_text(json.dumps(c))
 # Official converter concatenates paths, so convert absolute paths to relative paths.
 import os
 for o in c['objects'].values():o['mesh_path']=os.path.relpath(o['mesh_path'],d)
 Path(a['config']).write_text(json.dumps(c))
 a['reference_dir']=str(r/'results/detail256_reference'/a['name'])
 subprocess.run([str(r/'envs/renderformer/bin/python'),'scene_processor/to_blend.py',a['config'],'--output_dir',a['reference_dir'],'--resolution','256','--spp','2048','--save_img'],check=True)
(d/'manifest.json').write_text(json.dumps(m))
