"""One locked, validation-selected checkpoint evaluation on held-out test16."""
from head_data import *
import hashlib
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate import display_transform,measure,plt
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();RUN=Path(a.run)
result=json.loads((RUN/'result.json').read_text())
assert result['stop_reason']=='validation_plateau','Do not inspect test at epoch limit; training continuation requires validation-only decision'
best=RUN/'best.safetensors';checkpoint_sha=hashlib.sha256(best.read_bytes()).hexdigest()
assert checkpoint_sha==result['checkpoint_sha256']['best']
intent=RUN/'test-evaluation-intent.json'
locked_intent={'checkpoint_sha256':checkpoint_sha,'manifest_sha256':verify_manifest(),
 'initial_checkpoint_sha256':hashlib.sha256((RUN/'initial.safetensors').read_bytes()).hexdigest(),
 'policy':'One logical evaluation, same checkpoint/manifest only; resume missing outputs after interruption, never reselect'}
if intent.exists():
    assert json.loads(intent.read_text())==locked_intent,'Cannot resume with a different checkpoint or data'
else:intent.write_text(json.dumps(locked_intent,indent=2))
if (RUN/'test-summary.json').exists():raise SystemExit('Final test evaluation already complete')
m=json.loads((DATA/'manifest.json').read_text());frames=[f for f in m['frames'] if f['split']=='test'];assert len(frames)==16
items=load_items(frames);outputs={}
for label,path in [('pretrained',RUN/'initial.safetensors'),('finetuned',best)]:
    head=make_head(path);head.eval()
    with torch.no_grad():
        for item in items:
            folder=RUN/'test'/item['frame']['id'];folder.mkdir(parents=True,exist_ok=True)
            saved=folder/(label+'.exr');score=folder/(label+'-loss.json')
            if saved.exists() and score.exists():
                outputs.setdefault(item['frame']['id'],{})[label]={'image':iio.imread(saved),'log_mse':json.loads(score.read_text())['log_mse']}
                continue
            features,target=batch([item]);raw_log=predict_log(head,features);log=raw_log.float()
            image=(torch.pow(10.,raw_log)-1).float()[0].permute(1,2,0).cpu().numpy()
            mse=float(F.mse_loss(log,target));iio.imwrite(saved,image);score.write_text(json.dumps({'log_mse':mse}))
            outputs.setdefault(item['frame']['id'],{})[label]={'image':image,'log_mse':mse}
    del head;torch.cuda.empty_cache()
cases=[]
for item in items:
    f=item['frame'];folder=RUN/'test'/f['id'];folder.mkdir(parents=True,exist_ok=True)
    reference=iio.imread(DATA/f['reference'])[...,:3].astype(np.float32)
    metrics={}
    iio.imwrite(folder/'reference.png',np.round(display_transform(reference)*255).astype(np.uint8))
    for label in ['pretrained','finetuned']:
        record=outputs[f['id']][label];pred=record['image']
        assert np.isfinite(pred).all()
        iio.imwrite(folder/(label+'.exr'),pred)
        iio.imwrite(folder/(label+'.png'),np.round(display_transform(pred)*255).astype(np.uint8))
        metrics[label]=dict(measure(reference,pred),log_hdr_mse=record['log_mse'])
        error=np.mean(np.abs(display_transform(reference)-display_transform(pred)),axis=-1)
        plt.imsave(folder/(label+'-error.png'),error,cmap='magma',vmin=0,vmax=.25)
        metrics[label].update({'negative_prediction_fraction':float(np.mean(pred<0)),
          'error_saturated_pixel_fraction':float(np.mean(error>=.25))})
    row={'id':f['id'],'title':f['id'],'family':f['family'],'group_id':f['group_id'],'camera':f['camera'],
         'shape':list(reference.shape),'metrics':metrics,'label_stats':item['label_stats'],'output':str(folder),
         'error_scale':[0,.25],'error_note':'Display RGB MAE; values >=0.25 saturate at maximum magma color'}
    (folder/'metrics.json').write_text(json.dumps(row,indent=2));cases.append(row)
aggregate={}
for family in ['all','metal','sponza']:
    subset=[c for c in cases if family=='all' or c['family']==family]
    aggregate[family]={label:{key:float(np.mean([c['metrics'][label][key] for c in subset]))
        for key in ['log_hdr_mse','psnr_display_db','ssim_display','mae_linear_hdr','rmse_linear_hdr']}
        for label in ['pretrained','finetuned']}
summary={'test_evaluated':True,'checkpoint_sha256':checkpoint_sha,'test_frames':16,'test_cases':cases,'aggregate':aggregate,
 'aggregation':'Arithmetic mean of per-frame metrics; no image or family excluded',
 'display':'EV0; clamp nonnegative for display; channel-wise Reinhard; sRGB; all same 256px',
 'test_policy':'Only one validation-selected checkpoint, all 16 locked test frames; no hyperparameter selection'}
(RUN/'test-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(aggregate,indent=2))
(RUN/'state.json').write_text(json.dumps(dict(result,status='complete',test_evaluated=True,evaluated_checkpoint_sha256=checkpoint_sha),indent=2))
