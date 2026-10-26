"""One model load; capacity smoke tests on repeated fixtures, never a training run."""
from pilot_probe import *
import gc,statistics

out=Path('F:/RenderFormer-work/training-pilot/capacity');out.mkdir(parents=True,exist_ok=True)
pipe=RenderFormerRenderingPipeline.from_pretrained('F:/RenderFormer-work/models/renderformer-v1-base')
pipe.to(torch.device('cuda'));pipe.model.eval()
for p in pipe.model.parameters():p.requires_grad_(False)
head=pipe.model.view_transformer.out_dpt
for p in head.parameters():p.requires_grad_(True)
initial={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
records=[]
for family,fixture in [('metal',ROOT/'experiments/teapot/baseline'),('sponza',Path('F:/RenderFormer-work/sponza-architecture-experiment'))]:
    cpu=load_single_h5_data(str(fixture/'scene.h5'))
    raw=iio.imread(fixture/'reference.exr')[...,:3].copy()
    base_target=F.interpolate(torch.from_numpy(raw).permute(2,0,1)[None],size=(256,256),mode='area')[:,None]
    for batch in [1,2,4]:
        gc.collect();torch.cuda.empty_cache();head.load_state_dict(initial)
        row={'family':family,'resolution':256,'microbatch':batch,'status':'started'}
        try:
            data={k:v.unsqueeze(0).repeat(batch,*([1]*v.dim())).cuda() for k,v in cpu.items()}
            data['fov']=data['fov'].unsqueeze(-1)
            target=torch.log10(1+base_target.repeat(batch,1,1,1,1).cuda().clamp_min(0))
            optim=torch.optim.AdamW(head.parameters(),lr=1e-5,weight_decay=1e-4)
            scaler=torch.amp.GradScaler('cuda');times=[];losses=[];norms=[];free=[]
            torch.cuda.reset_peak_memory_stats()
            for step in range(3):
                optim.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
                pred=forward_log(pipe,data,256);loss=F.mse_loss(pred.float(),target)
                assert torch.isfinite(loss)
                scaler.scale(loss).backward();scaler.unscale_(optim)
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.0)
                assert torch.isfinite(norm) and norm>0
                scaler.step(optim);scaler.update();torch.cuda.synchronize()
                times.append(time.perf_counter()-start);losses.append(float(loss));norms.append(float(norm))
                free.append(torch.cuda.mem_get_info()[0])
            row.update(status='completed',losses=losses,gradient_norms=norms,step_seconds=times,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                min_free_device_bytes=min(free),warm_samples_per_second=batch/statistics.mean(times[1:]))
        except torch.cuda.OutOfMemoryError as error:
            row.update(status='out_of_memory',error=str(error)[:250])
        records.append(row)
        (out/'capacity.json').write_text(json.dumps({'records':records,'note':'Three steps per case, repeated smoke fixtures, no checkpoints saved. Throughput is indicative; no dataset/convergence claim.'},indent=2))
        print(json.dumps(row),flush=True)
        for name in ['data','target','optim','scaler','pred','loss']:
            globals().pop(name,None)
        for p in head.parameters():p.grad=None
        gc.collect();torch.cuda.empty_cache()
