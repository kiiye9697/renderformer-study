"""Prepare isolated paper-inspired camera/light stress tests; outputs are unmeasured.

Usage: python prepare_paper_badcases.py --official-repo /path/to/renderformer --output /path/to/cases
Run each config through the official converter, Cycles and inference before publishing results.
"""
import argparse
import copy
import json
import os
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--official-repo',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
base=json.loads((a.official_repo/'examples/cbox.json').read_text())
for obj in base['objects'].values():
    obj['mesh_path']=os.path.relpath(a.official_repo/'examples'/obj['mesh_path'],a.output)
cases=[]
def save(name,cfg,factor,figure,role='direct'):
    (a.output/(name+'.json')).write_text(json.dumps(cfg,indent=2))
    cases.append(dict(id=name,factor=factor,paper_figure=figure,role=role,status='prepared_not_run',resolution=512))
save('control',base,'baseline',13)
cfg=copy.deepcopy(base);cfg['cameras'][0]['position']=[0.,-0.2,0.]
save('camera_inside',cfg,'camera position only',13)
cfg=copy.deepcopy(base);cfg['objects']['light_0']['transform']['translation']=[0.,0.,0.5]
save('light_inside',cfg,'light position only',11)
cfg=copy.deepcopy(base);cfg['objects']['light_0']['transform']['scale']=[10.,10.,10.]
save('large_light',cfg,'emitter area; radiance held fixed, total power therefore changes',11)
cfg=copy.deepcopy(base);original=cfg['objects'].pop('light_0')
for i in range(9):
    light=copy.deepcopy(original)
    light['transform']['translation']=[(i%3-1)*.3,(i//3-1)*.3,2.1]
    light['transform']['scale']=[.6,.6,.6]
    cfg['objects'][f'light_{i}']=light
save('nine_lights_direct',cfg,'9 sources; compare with sum of matching single-source renders',10)
for i in range(9):
    single=copy.deepcopy(cfg)
    single['objects']={k:v for k,v in single['objects'].items() if not k.startswith('light_') or k==f'light_{i}'}
    save(f'nine_lights_component_{i}',single,'linear-HDR additive control',10,'component')
(a.output/'case_manifest.json').write_text(json.dumps(dict(cases=cases,
    rules=['Keep exact same mesh/camera on reference and prediction.',
           'Add component images in linear HDR before any tone mapping; never add display PNGs.',
           'Use matching full-9-light Cycles reference for direct and additive predictions.',
           'Higher-resolution and multi-bounce tests require separate configurations; not included here.',
           'These JSON files are configurations, not measured results.']),indent=2))
print('Prepared',len(cases),'unmeasured configs in',a.output)
