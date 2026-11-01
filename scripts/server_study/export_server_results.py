"""Export display images and measurements; never export assets or credentials."""
from pathlib import Path
import json,hashlib,os
import numpy as np
import imageio.v3 as iio
from skimage.metrics import structural_similarity

R=Path(os.environ['RENDERFORMER_WORKSPACE']);OUT=R/'results/public_export';OUT.mkdir(exist_ok=True)
def display(a):
    a=np.maximum(a,0);a=a/(1+a)
    return np.where(a<=.0031308,12.92*a,1.055*a**(1/2.4)-.055)
def write(path,a):
    path.parent.mkdir(parents=True,exist_ok=True);iio.imwrite(path,(np.clip(a,0,1)*255).astype(np.uint8))
def measures(a,b):
    x,y=display(a),display(b);mse=float(np.mean((x-y)**2))
    return dict(psnr_db=float(-10*np.log10(max(mse,1e-12))),ssim=float(structural_similarity(x,y,channel_axis=-1,data_range=1)),
                log_mse=float(np.mean((np.log10(1+np.maximum(a,0))-np.log10(1+np.maximum(b,0)))**2)))
def error(a,b):
    t=np.mean(np.abs(display(a)-display(b)),axis=-1)/.25
    return np.stack([np.clip(t*2,0,1),np.clip(t*2-1,0,1),np.zeros_like(t)],axis=-1)
run=R/'runs/bunny_sponza_decoder_fp32';manifest=json.loads((R/'datasets/training/manifest.json').read_text())
records=[]
for rec in manifest['records']:
    if rec['split']!='test':continue
    n=rec['name'];a=np.load(R/'results/decoder_adaptation/baseline'/n/'prediction_linear_hdr.npy')
    b=np.load(R/'results/decoder_adaptation/adapted'/n/'prediction_linear_hdr.npy')
    for view in range(3):
        ref=iio.imread(Path(rec['reference_dir'])/f'{n}_{view}.exr')[...,:3].astype(np.float32)
        row=dict(id=f'{rec["family"]}-{view}',family=rec['family'],view=view,
                 baseline=measures(a[view],ref),adapted=measures(b[view],ref))
        for key,img in [('reference',display(ref)),('baseline',display(a[view])),('adapted',display(b[view])),('baseline_error',error(a[view],ref)),('adapted_error',error(b[view],ref))]:
            name=f'training/{row["id"]}-{key}.png';write(OUT/name,img);row[key+'_image']=name
        records.append(row)
out=dict(code_commit='eec6a9b8d592303def45e458dd4ebd09d6572abc',model_revision='8aa0009563789dbb6c7785a26aac5da2c9b6d117',
         method='Frozen Transformer; FP32 DPT decoder adaptation',resolution=128,train_frames=18,validation_frames=6,test_frames=6,
         split='roughness 0.15/0.5/0.9 train, 0.3 validation, 0.7 test; 3 views per configuration; shared geometry and overlapping views',
         scope='material interpolation in two fixed scene families; not unseen-scene generalization',
         sponza_faces=manifest['sponza_reduced_faces'],sponza_limitations=manifest['sponza_limitations'],
         checkpoint_sha256=hashlib.sha256((run/'best_decoder.safetensors').read_bytes()).hexdigest(),
         history=json.loads((run/'history.json').read_text()),best_epoch=json.loads((run/'best_epoch.json').read_text()),
         display='EV0; clamp negative; channel-wise Reinhard; IEC sRGB',metric_domain='same float display RGB as images, data_range=1; log MSE on linear HDR',
         error_scale=[0,.25],error_colormap='black-red-yellow, threshold 0.25',cases=records)
(OUT/'server-training.json').write_text(json.dumps(out,indent=2))
bad=[];base=R/'results/paper_badcases'
for name in ['control','camera_inside','light_inside','large_light','nine_lights_direct']:
    pred=base/name/'prediction'/f'{name}_view_0.exr';ref=base/name/'reference'/f'{name}_0.exr'
    if not pred.exists() or not ref.exists():continue
    a=iio.imread(pred)[...,:3].astype(np.float32);b=iio.imread(ref)[...,:3].astype(np.float32)
    row=dict(id=name,metrics=measures(a,b),status='measured')
    for key,img in [('reference',display(b)),('prediction',display(a)),('error',error(a,b))]:
        path=f'badcases/{name}-{key}.png';write(OUT/path,img);row[key]=path
    bad.append(row)
parts=[base/f'nine_lights_component_{i}'/'prediction'/f'nine_lights_component_{i}_view_0.exr' for i in range(9)]
if all(x.exists() for x in parts):
    a=sum(iio.imread(x)[...,:3].astype(np.float32) for x in parts)
    b=iio.imread(base/'nine_lights_direct/reference/nine_lights_direct_0.exr')[...,:3].astype(np.float32)
    row=dict(id='nine_lights_sum',metrics=measures(a,b),status='measured')
    for key,img in [('reference',display(b)),('prediction',display(a)),('error',error(a,b))]:
        path=f'badcases/nine_lights_sum-{key}.png';write(OUT/path,img);row[key]=path
    bad.append(row)
(OUT/'server-badcases.json').write_text(json.dumps(dict(resolution=512,reference_spp=2048,display=out['display'],metric_domain=out['metric_domain'],error_scale=[0,.25],cases=bad),indent=2))
print('Exported',len(records),'training test views and',len(bad),'badcase measurements')
