"""Freeze 40 complete configuration groups / 80 frames before training.
No images are inspected and no split is selected using model performance.
"""
from pathlib import Path
import json,copy,random,hashlib,math,os
ROOT=Path(__file__).resolve().parents[2]
OUT=Path('F:/RenderFormer-work/training-pilot/dataset-v1')
SEED=20261027
rng=random.Random(SEED)
if (OUT/'manifest.json').exists():
    raise SystemExit('Manifest already locked; refusing to regenerate dataset-v1')
OUT.mkdir(parents=True,exist_ok=True)
sources={'metal':ROOT/'experiments/teapot/baseline/config.json',
         'sponza':Path('F:/RenderFormer-work/sponza-architecture-experiment/config.json')}
frames=[];groups=[]
for family,source in sources.items():
    base=json.loads(source.read_text())
    for obj in base['objects'].values():
        obj['mesh_path']=str((source.parent/obj['mesh_path']).resolve())
    ids=list(range(20));rng.shuffle(ids)
    split_of={g:('train' if k<12 else 'val' if k<16 else 'test') for k,g in enumerate(ids)}
    for gid in range(20):
        split=split_of[gid];group_id=f'{family}-g{gid:02d}'
        scene=copy.deepcopy(base)
        scene['scene_name']=group_id
        if family=='metal':
            scene['objects']['teapot']['material']['roughness']=rng.uniform(.12,.8)
            scene['objects']['teapot']['transform']['rotation'][2]=rng.uniform(-50,65)
            scene['objects']['background_3']['material']['diffuse']=[rng.uniform(.1,.65),0.,0.]
            scene['objects']['background_2']['material']['diffuse']=[0.,rng.uniform(.1,.65),0.]
        else:
            scale=rng.uniform(.75,1.15);rough=rng.uniform(.55,.9)
            for name,obj in scene['objects'].items():
                if name=='light_0':continue
                obj['material']['diffuse']=[min(.85,x*scale) for x in obj['material']['diffuse']]
                obj['material']['roughness']=rough
        lamp=scene['objects']['light_0']
        lamp['material']['emissive']=[rng.uniform(3500,6000)]*3
        lamp['transform']['translation']=[rng.uniform(-.3,.3),rng.uniform(-.3,.3),2.1]
        common=copy.deepcopy(scene);common.pop('cameras')
        common_hash=hashlib.sha256(json.dumps(common,sort_keys=True).encode()).hexdigest()
        groups.append({'id':group_id,'family':family,'split':split,'configuration_sha256':common_hash})
        for view in range(2):
            config=copy.deepcopy(scene);frame_id=f'{group_id}-v{view}'
            if family=='metal':
                lo,hi={'train':(-12,12),'val':(-28,-20),'test':(20,28)}[split]
                angle=math.radians(rng.uniform(lo,hi));dist=rng.uniform(1.8,2.)
                camera={'position':[dist*math.sin(angle),-dist*math.cos(angle),rng.uniform(-.08,.08)],
                        'look_at':[0.,0.,-.05],'up':[0.,0.,1.],'fov':40.}
            else:
                lo,hi={'train':(-.02,.02),'val':(-.09,-.06),'test':(.06,.09)}[split]
                camera={'position':[rng.uniform(lo,hi),rng.uniform(-.42,-.35) if view==0 else rng.uniform(-.30,-.24),rng.uniform(-.17,-.11)],
                        'look_at':[0.,.25,-.08],'up':[0.,0.,1.],'fov':rng.uniform(58,64)}
            config['cameras']=[camera]
            folder=OUT/'frames'/frame_id;folder.mkdir(parents=True)
            # Paths are local-only; source assets are never copied or published.
            for obj in config['objects'].values():obj['mesh_path']=os.path.relpath(obj['mesh_path'],folder).replace('\\','/')
            data=json.dumps(config,indent=2)
            (folder/'config.json').write_bytes(data.encode())
            frames.append({'id':frame_id,'group_id':group_id,'family':family,'split':split,
                'config':f'frames/{frame_id}/config.json','h5':f'frames/{frame_id}/scene.h5',
                'reference':f'frames/{frame_id}/reference.exr','config_sha256':hashlib.sha256(data.encode()).hexdigest(),
                'camera':camera})
manifest={'version':1,'seed':SEED,'locked_before_training':True,'resolution':256,'cycles_spp':2048,'cycles_seed':11,
 'groups':groups,'frames':frames,'counts':{s:sum(f['split']==s for f in frames) for s in ['train','val','test']},
 'split_policy':'Complete geometry/material/light configuration groups remain in one split; two views per group. Camera sectors are split-disjoint. Same fixed scene families: adaptation, not novel-scene generalization.',
 'test_policy':'Test16 reserved for one final evaluation of validation-selected checkpoint; never select hyperparameters/checkpoint using test or historical8.',
 'camera_blocks':{'metal_azimuth_degrees':{'train':[-12,12],'val':[-28,-20],'test':[20,28]},'sponza_camera_x':{'train':[-.02,.02],'val':[-.09,-.06],'test':[.06,.09]}},
 'historical_fixture_policy':'Prior8 images used only for capacity smoke tests and final historical regression; not part of this dataset.'}
assert manifest['counts']=={'train':48,'val':16,'test':16}
assert all(len({f['split'] for f in frames if f['group_id']==g['id']})==1 for g in groups)
text=json.dumps(manifest,indent=2);(OUT/'manifest.json').write_bytes(text.encode())
(OUT/'manifest.sha256').write_text(hashlib.sha256(text.encode()).hexdigest()+'\n')
print('Locked',OUT/'manifest.json',manifest['counts'])
