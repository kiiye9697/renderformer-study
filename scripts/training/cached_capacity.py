"""Measure head-only microbatches on train inputs; never evaluate test."""
from head_data import *
import gc
m=json.loads((DATA/'manifest.json').read_text())
items=load_items([f for f in m['frames'] if f['split']=='train'][:8])
def probe(size):
    head=make_head();head.train();optim=torch.optim.AdamW(head.parameters(),lr=1e-5,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda');features,target=batch(items[:size]);torch.cuda.reset_peak_memory_stats();times=[]
    for step in range(4):
        torch.cuda.synchronize();start=time.perf_counter();optim.zero_grad(set_to_none=True)
        loss=F.mse_loss(predict_log(head,features).float(),target)
        scaler.scale(loss).backward();scaler.unscale_(optim);torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        scaler.step(optim);scaler.update();torch.cuda.synchronize();times.append(time.perf_counter()-start)
    free,total=torch.cuda.mem_get_info()
    record={'microbatch':size,'status':'completed','seconds':times,'warm_samples_per_second':size/(sum(times[1:])/3),
       'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
       'free_bytes':free,'total_bytes':total,'last_loss':float(loss),'disposable_probe_weights':True,
       'timing_scope':'CUDA-synchronized head forward, loss, backward, clipped optimizer step; features already resident on GPU; excludes CPU loading/transfer and model initialization'}
    del head,optim,scaler,features,target,loss;gc.collect();torch.cuda.empty_cache()
    return record
records=[]
for size in [4,8]:
    try:records.append(probe(size))
    except torch.cuda.OutOfMemoryError as exc:
        records.append({'microbatch':size,'status':'out_of_memory','error':str(exc),'free_bytes':0,'warm_samples_per_second':0.})
    gc.collect();torch.cuda.empty_cache()
if all(r['status']=='completed' for r in records):
    rates=[r['warm_samples_per_second'] for r in records]
    if max(rates)/min(rates)<1.1:
        for record in reversed(records):
            try:extra=probe(record['microbatch'])
            except torch.cuda.OutOfMemoryError as exc:
                record.update(status='out_of_memory_on_repeat',free_bytes=0,repeat_error=str(exc))
                gc.collect();torch.cuda.empty_cache();continue
            record['repeat_reverse_order']=extra
            times=record['seconds'][1:]+extra['seconds'][1:]
            record['warm_samples_per_second']=record['microbatch']/(sum(times)/len(times))
            record['free_bytes']=min(record['free_bytes'],extra['free_bytes'])
            record['peak_reserved_bytes']=max(record['peak_reserved_bytes'],extra['peak_reserved_bytes'])
            record['peak_allocated_bytes']=max(record['peak_allocated_bytes'],extra['peak_allocated_bytes'])
out=Path('F:/RenderFormer-work/training-pilot/cached-capacity.json');out.write_text(json.dumps(records,indent=2))
print(out.read_text(),flush=True)
