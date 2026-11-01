import os
import json,subprocess
from pathlib import Path
r=Path(os.environ['RENDERFORMER_WORKSPACE']);p=r/'datasets/detail256/manifest.json';m=json.loads(p.read_text())
for a in m['records']:
 if a['split']=='test':
  a['h5']=str(r/'datasets/detail256'/(a['name']+'.h5'))
  subprocess.run([str(r/'envs/renderformer/bin/python'),'scene_processor/convert_scene.py',a['config'],'--output_h5_path',a['h5']],check=True)
p.write_text(json.dumps(m))
