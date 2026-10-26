"""Run only after approved data/cache audit. Validation selects; test is never read."""
from head_data import *
import random,hashlib
p=argparse.ArgumentParser();p.add_argument('--microbatch',type=int,choices=[4,8],required=True);p.add_argument('--output',required=True)
a=p.parse_args();RUN=Path(a.output)
if RUN.exists():raise SystemExit('Refusing to overwrite an existing training run')
RUN.mkdir(parents=True)
torch.manual_seed(20261028);rng=random.Random(20261028)
m=json.loads((DATA/'manifest.json').read_text())
train=load_items([f for f in m['frames'] if f['split']=='train'])
val=load_items([f for f in m['frames'] if f['split']=='val'])
assert len(train)==48 and len(val)==16
head=make_head();save_head(head,RUN/'initial.safetensors');save_head(head,RUN/'best.safetensors')
optim=torch.optim.AdamW(head.parameters(),lr=1e-5,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda')
config={'microbatch':a.microbatch,'gradient_accumulation':1,'effective_batch':a.microbatch,
 'max_epochs':100,'patience':12,'relative_min_delta':.002,'lr':1e-5,'weight_decay':1e-4,'optimizer':'AdamW',
 'loss':'MSE(log10(1+linear HDR)), mean over pixels/channels/frames','clip_grad_norm':1.,'seed':20261028,
 'train_frames':48,'val_frames':16,'test_frames':16,'trainable_parameters':sum(p.numel() for p in head.parameters()),
 'frozen_backbone':'all parameters except DPT output head; features cached once with original dtype',
 'cached_feature_dtypes':[str(x.dtype) for x in train[0]['features']],
 'manifest_sha256':(DATA/'manifest.sha256').read_text().strip(),'test_evaluated':False}
config.update({'source_model_sha256':MODEL_SHA,'official_code_commit':'eec6a9b8d592303def45e458dd4ebd09d6572abc',
 'torch_version':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
 'script_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
    for name in ['train_head.py','head_data.py','cache_features.py']}})
(RUN/'config.json').write_text(json.dumps(config,indent=2))
(RUN/'label-stats.json').write_text(json.dumps({x['frame']['id']:x['label_stats'] for x in train+val},indent=2))
v=evaluate_loss(head,val,a.microbatch);t=evaluate_loss(head,train,a.microbatch)
history=[{'epoch':0,'step':0,'train_loss':t['mean'],'val_loss':v['mean'],'val_by_family':v['families']}]
best=v['mean'];patience_anchor=best;best_epoch=0;bad=0;step=0;reason='max_epochs';start_all=time.perf_counter()
for epoch in range(1,101):
    order=list(range(len(train)));rng.shuffle(order);head.train();losses=[]
    for j in range(0,len(order),a.microbatch):
        part=[train[i] for i in order[j:j+a.microbatch]];features,target=batch(part)
        optim.zero_grad(set_to_none=True);loss=F.mse_loss(predict_log(head,features).float(),target)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss; no convergence claim')
        scaler.scale(loss).backward();scaler.unscale_(optim)
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('Non-finite gradient')
        scaler.step(optim);scaler.update();step+=1;losses.append(float(loss))
        with (RUN/'steps.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch,'step':step,'train_loss':float(loss),'grad_norm':float(norm)})+'\n')
    v=evaluate_loss(head,val,a.microbatch);t=evaluate_loss(head,train,a.microbatch)
    if v['mean']<best:
        best=v['mean'];best_epoch=epoch;save_head(head,RUN/'best.safetensors')
    if v['mean']<patience_anchor*(1-.002):
        patience_anchor=v['mean'];bad=0
    else:bad+=1
    save_head(head,RUN/'last.safetensors')
    row={'epoch':epoch,'step':step,'train_loss':t['mean'],'online_train_loss':sum(losses)/len(losses),'val_loss':v['mean'],'val_by_family':v['families'],
         'best_epoch':best_epoch,'patience_used':bad,'elapsed_seconds':time.perf_counter()-start_all}
    history.append(row)
    (RUN/'history.json').write_text(json.dumps(history,indent=2))
    (RUN/'state.json').write_text(json.dumps({'status':'running','epoch':epoch,'step':step,'best_epoch':best_epoch,'best_val_loss':best,'test_evaluated':False},indent=2))
    print(json.dumps(row),flush=True)
    if bad>=12:reason='validation_plateau';break
result={'status':'training_finished_test_pending','stop_reason':reason,'best_epoch':best_epoch,'best_val_loss':best,
        'epochs':epoch,'optimizer_steps':step,'elapsed_seconds':time.perf_counter()-start_all,'test_evaluated':False,
        'convergence_interpretation':'Operational validation plateau only' if reason=='validation_plateau' else 'Epoch limit reached; convergence not established',
        'timing_scope':'Epoch loop including CPU batch transfer, updates, epoch-end train/val evaluation and head saves; excludes dataset generation, feature cache, initial loading/evaluation and final optimizer-state serialization',
        'checkpoint_sha256':{x:hashlib.sha256((RUN/(x+'.safetensors')).read_bytes()).hexdigest() for x in ['initial','best','last']}}
torch.save({'optimizer':optim.state_dict(),'scaler':scaler.state_dict(),'python_rng':rng.getstate(),
 'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'epoch':epoch,'step':step,
 'best_val_loss':best,'best_epoch':best_epoch,'patience_anchor':patience_anchor,'patience_used':bad},RUN/'last-training-state.pt')
(RUN/'state.json').write_text(json.dumps(result,indent=2))
(RUN/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
