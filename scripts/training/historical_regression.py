"""Final-only historical 512px regression; not validation or checkpoint selection."""
from pilot_probe import *
from safetensors.torch import load_file
import shutil,hashlib
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate import measure,display_transform,plt
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();RUN=Path(a.run)
assert (RUN/'test-summary.json').exists()
best_sha=hashlib.sha256((RUN/'best.safetensors').read_bytes()).hexdigest()
assert best_sha==json.loads((RUN/'result.json').read_text())['checkpoint_sha256']['best']
pipe=RenderFormerRenderingPipeline.from_pretrained('F:/RenderFormer-work/models/renderformer-v1-base');pipe.to(torch.device('cuda'))
pipe.model.view_transformer.out_dpt.load_state_dict(load_file(str(RUN/'best.safetensors')));pipe.model.eval()
cases=[('cbox',ROOT/'experiments/cbox','reference_4096.exr')]
cases += [('metal_'+name,ROOT/'experiments/teapot'/name,'reference.exr') for name in ['baseline','roughness_015','roughness_075','wall_000','wall_050']]
cases += [('sponza_interior',Path('F:/RenderFormer-work/sponza-architecture-experiment'),'reference.exr'),('sponza_exterior',Path('F:/RenderFormer-work/sponza-exterior-diagnostic'),'reference.exr')]
rows=[];assets=RUN/'publication/assets/regression';assets.mkdir(parents=True,exist_ok=True)
for name,source,ref in cases:
    folder=RUN/'historical-regression'/name;folder.mkdir(parents=True,exist_ok=True)
    data={k:v.unsqueeze(0).cuda() for k,v in load_single_h5_data(str(source/'scene.h5')).items()};data['fov']=data['fov'].unsqueeze(-1)
    with torch.no_grad():pred=pipe(**data,resolution=512,torch_dtype=torch.float16)[0,0].float().cpu().numpy()
    assert pred.shape==(512,512,3) and np.isfinite(pred).all()
    iio.imwrite(folder/'finetuned.exr',pred)
    reference=iio.imread(source/ref)[...,:3];pretrained=iio.imread(source/'prediction/prediction.exr')[...,:3]
    row={'id':name,'family':'metal' if name.startswith('metal') else 'sponza' if name.startswith('sponza') else 'cbox',
      'resolution':512,'metrics':{'pretrained':measure(reference,pretrained),'finetuned':measure(reference,pred)}}
    for key,image in [('reference',reference),('pretrained',pretrained),('finetuned',pred)]:
        filename=name+'-'+key+'.png';iio.imwrite(assets/filename,np.round(display_transform(image)*255).astype(np.uint8));row[key]='assets/regression/'+filename
        if key!='reference':
            error=np.mean(np.abs(display_transform(reference)-display_transform(image)),axis=-1)
            filename=name+'-'+key+'-error.png';plt.imsave(assets/filename,error,cmap='magma',vmin=0,vmax=.25);row[key+'_error']='assets/regression/'+filename
            row['metrics'][key].update(negative_prediction_fraction=float(np.mean(image<0)),error_saturated_pixel_fraction=float(np.mean(error>=.25)))
    row.update(error_scale=[0,.25],error_note='Display RGB MAE; values >=0.25 saturate at maximum magma color')
    rows.append(row);del data
summary={'purpose':'Final-only historical 8-case regression, original512px; not pooled with new256px test and never used for selection',
 'cases':rows,'checkpoint_sha256':best_sha,'labels':'A=Cycles reference; B=original pretrained; C=validation-selected finetuned'}
(RUN/'publication/historical-regression.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
