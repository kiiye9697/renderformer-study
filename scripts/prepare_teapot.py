"""Create controlled configurations sharing one pre-remeshed teapot."""
from pathlib import Path
import json, copy, os
ROOT=Path(__file__).resolve().parents[1]
official=ROOT/'vendor/renderformer/examples'
base=json.loads((official/'cbox.json').read_text())
del base['objects']['tall_box']; del base['objects']['short_box']
for obj in base['objects'].values():
    obj['mesh_path']=str((official/obj['mesh_path']).resolve())
base['objects']['teapot']={
 'mesh_path':str(ROOT/'experiments/teapot/geometry/teapot_2048.obj'),
 'transform':{'translation':[0.,0.,-.5],'rotation':[0.,0.,25.],'scale':[.38,.38,.38],'normalize':False},
 'material':{'diffuse':[0.,0.,0.],'specular':[.9,.9,.9],'roughness':.4,'emissive':[0.,0.,0.],'smooth_shading':True},
 'remesh':False}
cases=[('roughness_015',.15,1.),('baseline',.4,1.),('roughness_075',.75,1.),('wall_000',.4,0.),('wall_050',.4,.5)]
manifest=[]
for name,rough,wall in cases:
    d=copy.deepcopy(base); d['scene_name']='Cornell box + Utah teapot: '+name
    d['objects']['teapot']['material']['roughness']=rough
    d['objects']['background_3']['material']['diffuse']=[.5*wall,0.,0.]
    folder=ROOT/'experiments/teapot'/name; folder.mkdir(parents=True,exist_ok=True)
    for obj in d['objects'].values(): obj['mesh_path']=os.path.relpath(obj['mesh_path'],folder).replace('\\','/')
    (folder/'config.json').write_text(json.dumps(d,indent=2))
    manifest.append({'id':name,'roughness':rough,'red_wall_strength':wall,'status':'pending'})
(ROOT/'experiments/teapot/design.json').write_text(json.dumps({'cases':manifest,
 'material':'achromatic metal approximation: diffuse=[0,0,0], specular=[0.9,0.9,0.9]; official additive GGX shader, not Unity metallic workflow',
 'geometry':'shared official remesh, target 2048 faces; no per-case remesh',
 'roi_teapot':[180,330,330,400], 'roi_floor':[100,460,180,490],
 'roi_format':'[x0,y0,x1,y1], selected before viewing model predictions',
 'evaluation':'512x512, linear HDR preserved; fixed EV=0/Reinhard/sRGB; full-image metrics and fixed ROI means'},indent=2))
