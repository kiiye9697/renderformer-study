"""Create sanitized publication staging, never publish or copy mesh/weight assets."""
from pathlib import Path
import argparse,json,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();RUN=Path(a.run)
OUT=RUN/'publication';ASSETS=OUT/'assets/training';ASSETS.mkdir(parents=True,exist_ok=True)
config=json.loads((RUN/'config.json').read_text());history=json.loads((RUN/'history.json').read_text())
result=json.loads((RUN/'result.json').read_text());test=json.loads((RUN/'test-summary.json').read_text())
cases=[]
for c in test['test_cases']:
    row={k:c[k] for k in ['id','title','family','group_id','camera','shape','metrics','error_scale','error_note']}
    for key in ['reference','pretrained','finetuned']:
        filename=c['id']+'-'+key+'.png';shutil.copy2(Path(c['output'])/(key+'.png'),ASSETS/filename)
        row[key]='assets/training/'+filename
    for key in ['pretrained','finetuned']:
        filename=c['id']+'-'+key+'-error.png';shutil.copy2(Path(c['output'])/(key+'-error.png'),ASSETS/filename)
        row[key+'_error']='assets/training/'+filename
    cases.append(row)
fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
for key,label in [('train_loss','Training (epoch-end eval)'),('val_loss','Validation')]:
    ax.plot([r['epoch'] for r in history],[r[key] for r in history],label=label)
ax.axvline(result['best_epoch'],ls='--',color='grey',label=f"Selected checkpoint (epoch {result['best_epoch']})")
ax.set(xlabel='Epoch',ylabel=r'MSE in $\log_{10}(1+I_{\mathrm{HDR}})$',yscale='log');ax.legend();ax.grid(alpha=.2)
fig.savefig(ASSETS/'loss-curve.png',dpi=160);plt.close(fig)
summary={'status':'complete','config':dict({k:v for k,v in config.items() if k!='test_evaluated'},micro_batch=config['microbatch'],epochs=result['epochs'],optimizer_steps=result['optimizer_steps']),
 'batch_capacity':json.loads((RUN.parent/'cached-capacity.json').read_text()),
 'batch_choice':json.loads((RUN.parent/'batch-choice.json').read_text()),
 'history':history,'test_evaluated':True,'test_cases':cases,'aggregate':test['aggregate'],
 'loss_curve':'assets/training/loss-curve.png','loss_definition':config['loss'],
 'checkpoint':{'sha256':result['checkpoint_sha256']['best'],'best_epoch':result['best_epoch'],'best_val_loss':result['best_val_loss']},
 'selection_rule':'Minimum validation loss; independent 0.2% improvement anchor and patience12 stopping; test16 never selects model',
 'stop_reason':result['stop_reason'],'convergence_interpretation':result['convergence_interpretation'],
 'provenance':{'training_scope':'DPT output head only; frozen pretrained scene/view transformer; same two scene families',
   'data':'40 configuration groups, 2 camera views each, split 12/4/4 groups per family; 48/16/16 frames',
   'reference':'256x256 Cycles 2048spp seed11, same converted geometry/normals/materials/camera, no denoise',
   'labels':'A=Cycles reference; B=pretrained; C=finetuned',
   'display':test['display'],'test_policy':test['test_policy'],'timing_seconds':result['elapsed_seconds'],
   'timing_scope':result['timing_scope']}}
(OUT/'training-summary.json').write_text(json.dumps(summary,indent=2))
print(OUT)
