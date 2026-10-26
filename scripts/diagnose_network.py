"""Separate real-forward diagnostic; statistics are activations, not physical light transport."""
from pathlib import Path
import os,sys,json,argparse
os.environ['ATTN_IMPL']='sdpa'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/renderformer'))
import torch
from infer import load_single_h5_data
from renderformer import RenderFormerRenderingPipeline
p=argparse.ArgumentParser();p.add_argument('--h5',required=True);p.add_argument('--output',required=True)
a=p.parse_args()
model_path='F:/RenderFormer-work/models/renderformer-v1-base'
pipe=RenderFormerRenderingPipeline.from_pretrained(model_path);pipe.to(torch.device('cuda'))
records=[]
def describe(v):
    if isinstance(v,torch.Tensor):
        z=v.detach().float()
        return {'shape':list(v.shape),'dtype':str(v.dtype),'mean':float(z.mean()),'std':float(z.std()),'min':float(z.min()),'max':float(z.max())}
    if isinstance(v,(tuple,list)):return [describe(x) for x in v]
    if isinstance(v,dict):return {k:describe(x) for k,x in v.items()}
    return str(type(v).__name__)
names=['texture_encoder','vn_encoding_proj','transformer','view_transformer.ray_map_encoder',
       'view_transformer.transformer','view_transformer.out_dpt','view_transformer']
for name,module in pipe.model.named_modules():
    if name in names:
        def hook(m,inputs,kwargs,outputs,name=name):
            records.append({'module':name,'type':type(m).__name__,'inputs':describe(inputs),'keyword_inputs':describe(kwargs),'outputs':describe(outputs)})
        module.register_forward_hook(hook,with_kwargs=True)
data={k:v.unsqueeze(0).cuda() for k,v in load_single_h5_data(a.h5).items()};data['fov']=data['fov'].unsqueeze(-1)
result=pipe(**data,resolution=512,torch_dtype=torch.float16)
Path(a.output).write_text(json.dumps({'status':'measured','source_h5':a.h5,'resolution':512,
 'model_id':'microsoft/renderformer-v1-base','revision':'8aa0009563789dbb6c7785a26aac5da2c9b6d117',
 'config':json.loads(Path(model_path,'config.json').read_text()),'records':records,'linear_hdr_output':describe(result),
 'warning':'Separate hooked inference excluded from performance measurements. Activation statistics are not attention maps or physical irradiance.'},indent=2))
print('Saved',a.output)
