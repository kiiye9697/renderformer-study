"""Three-step viability probe only. No trained weights are persisted.

Uses an existing scene as a smoke-test fixture, not as a training/validation result.
"""
from pathlib import Path
import os,sys,json,time,argparse
os.environ['ATTN_IMPL']='sdpa'
os.environ.setdefault('IMAGEIO_FREEIMAGE_LIB','F:/RenderFormer-work/imageio/freeimage-3.18.0-win64.dll')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'vendor/renderformer'))
import torch,numpy as np,imageio.v3 as iio
import torch.nn.functional as F
from infer import load_single_h5_data
from renderformer import RenderFormerRenderingPipeline
from renderformer.utils.transform import trans_to_cam_coord

def forward_log(pipe,data,resolution):
    """Mirror public pipeline preprocessing; preserve gradients through model only."""
    triangles,texture,mask,vn,c2w,fov=[data[k] for k in ['triangles','texture','mask','vn','c2w','fov']]
    texture=texture.clone()
    if pipe.config.texture_encode_patch_size==1 and texture.dim()==5:
        texture=texture[:,:,:,0,0]
    if not pipe.config.use_ldr:
        texture[:,:,-3:]=torch.log10(texture[:,:,-3:]+1)
    bs,nv=c2w.shape[:2]
    if pipe.config.turn_to_cam_coord:
        tris,cam,_=trans_to_cam_coord(c2w.reshape(-1,4,4),triangles.repeat_interleave(nv,dim=0))
        tris=tris.reshape(bs,nv,-1,3,3);cam=cam.reshape(bs,nv,4,4)
    else:
        tris=triangles.unsqueeze(1).expand(-1,nv,-1,-1,-1);cam=c2w
    rays_o,rays_d=pipe.ray_generator(cam,fov/180*torch.pi,resolution)
    with torch.autocast('cuda',dtype=torch.float16):
        return pipe.model(triangles.reshape(bs,-1,9),texture,mask,vn.reshape(bs,-1,9),
            rays_o=rays_o,rays_d=rays_d,tri_vpos_view_tf=tris.reshape(bs,nv,-1,9),tf32_view_tf=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--resolution',type=int,required=True)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--fixture',choices=['metal','sponza'],default='metal')
    p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(20261027)
    pipe=RenderFormerRenderingPipeline.from_pretrained('F:/RenderFormer-work/models/renderformer-v1-base')
    pipe.to(torch.device('cuda'));pipe.model.eval()
    for par in pipe.model.parameters():par.requires_grad_(False)
    head=pipe.model.view_transformer.out_dpt
    for par in head.parameters():par.requires_grad_(True)
    fixture=ROOT/'experiments/teapot/baseline' if a.fixture=='metal' else Path('F:/RenderFormer-work/sponza-architecture-experiment')
    data={k:v.unsqueeze(0).cuda() for k,v in load_single_h5_data(str(fixture/'scene.h5')).items()}
    data={k:v.repeat(a.batch,*([1]*(v.dim()-1))) for k,v in data.items()}
    data['fov']=data['fov'].unsqueeze(-1)
    raw=iio.imread(fixture/'reference.exr')[...,:3].copy()
    target=torch.from_numpy(raw).permute(2,0,1)[None].cuda()
    target=F.interpolate(target,size=(a.resolution,a.resolution),mode='area')[:,None]
    target=target.repeat(a.batch,1,1,1,1)
    target_log=torch.log10(1+target.clamp_min(0))
    with torch.no_grad():
        official=pipe(**{k:v.clone() for k,v in data.items()},resolution=a.resolution,torch_dtype=torch.float16)
        pred=forward_log(pipe,data,a.resolution)
        hdr=(torch.pow(10.,pred)-1).permute(0,1,3,4,2)
        parity=float((official-hdr).abs().max())
        initial_loss=float(F.mse_loss(pred.float(),target_log))
    assert parity<1e-5,parity
    optim=torch.optim.AdamW(head.parameters(),lr=1e-5,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda')
    first_param=next(head.parameters());first_before=first_param.detach().clone()
    times=[];losses=[];grad_norms=[]
    torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        optim.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
        pred=forward_log(pipe,data,a.resolution)
        loss=F.mse_loss(pred.float(),target_log)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.unscale_(optim)
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.0)
        assert torch.isfinite(norm) and norm>0
        scaler.step(optim);scaler.update();torch.cuda.synchronize()
        times.append(time.perf_counter()-start);losses.append(float(loss));grad_norms.append(float(norm))
    with torch.no_grad():
        final_loss=float(F.mse_loss(forward_log(pipe,data,a.resolution).float(),target_log))
    frozen_grad_count=sum(p.grad is not None for p in pipe.model.parameters() if not p.requires_grad)
    result={'status':'probe_completed_not_training_result','resolution':a.resolution,'microbatch':a.batch,'fixture':a.fixture,
      'optimizer':'AdamW','lr':1e-5,'steps':3,'loss':'MSE(log10(1+linear HDR))',
      'total_parameters':sum(p.numel() for p in pipe.model.parameters()),
      'trainable_parameters':sum(p.numel() for p in head.parameters()),
      'trainable_module':'view_transformer.out_dpt','initial_loss':initial_loss,'step_losses':losses,
      'final_loss':final_loss,'gradient_norm_before_clip':grad_norms,'step_seconds':times,
      'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
      'official_pipeline_parity_max_abs':parity,'frozen_parameters_with_gradient':frozen_grad_count,
      'first_head_parameter_max_change':float((first_param-first_before).abs().max()),
      'fixture_note':'Existing scene used exclusively for execution smoke test; larger batches repeat it for capacity measurement, not data diversity. Target area-resized from 512 HDR; fresh training references must be rendered at training resolution. No checkpoint saved; no generalization or convergence claim.'}
    (out/'probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
