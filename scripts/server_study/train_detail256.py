"""Fine-tune only the official DPT image decoder, with frozen Transformer features.

This is a small supervised adaptation experiment, not original-scale training.
"""
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v3 as iio
from safetensors.torch import save_file
from skimage.metrics import structural_similarity
from renderformer import RenderFormerRenderingPipeline
from infer import load_single_h5_data

ROOT = Path(os.environ['RENDERFORMER_WORKSPACE'])
RUN = ROOT / 'runs/sponza_detail_256'
RUN.mkdir(exist_ok=True)
torch.manual_seed(20261101)
np.random.seed(20261101)
manifest = json.loads((ROOT / 'datasets/detail256/manifest.json').read_text())
resolution = manifest['resolution']
pipeline = RenderFormerRenderingPipeline.from_pretrained(str(ROOT / 'models/renderformer-v1-base-8aa00095'))
pipeline.to(torch.device('cuda'))
for parameter in pipeline.model.parameters():
    parameter.requires_grad_(False)
head = pipeline.model.view_transformer.out_dpt
for parameter in head.parameters():
    parameter.requires_grad_(True)
print('Trainable decoder parameters:', sum(p.numel() for p in head.parameters()), flush=True)
cache = []
captured = []
def capture(module, inputs):
    captured.append(([x[0].detach().cpu().float() for x in inputs[0]], inputs[1:]))
handle = head.register_forward_pre_hook(capture)
for record in ([] if cache else manifest['records']):
    data = load_single_h5_data(record['h5'])
    captured.clear()
    output = pipeline(triangles=data['triangles'][None].cuda(), texture=data['texture'][None].cuda(),
        mask=data['mask'][None].cuda(), vn=data['vn'][None].cuda(),
        c2w=data['c2w'][None].cuda(), fov=data['fov'][None, :, None].cuda(),
        resolution=resolution, torch_dtype=torch.float16)
    assert torch.isfinite(output).all(), record['name']
    features, shape = captured[0]
    targets = []
    for view in range(data['c2w'].shape[0]):
        path = Path(record['reference_dir']) / f"{record['name']}_{view}.exr"
        rgb = iio.imread(path)[..., :3].astype(np.float32)
        assert np.isfinite(rgb).all(), str(path)
        targets.append(np.log10(1. + np.maximum(rgb, 0.)))
    target = torch.from_numpy(np.stack(targets)).permute(0, 3, 1, 2)
    cache.append(dict(record=record, features=features, shape=shape, target=target))
    print('Cached', record['name'], 'features', [tuple(x.shape) for x in features], flush=True)
handle.remove()
torch.save(cache, RUN / 'frozen_features.pt')
head.train()
optimizer = torch.optim.AdamW(head.parameters(), lr=5e-6, weight_decay=1e-4)
scaler = torch.amp.GradScaler('cuda', enabled=False)
def detail_loss(pred, target):
    base = F.l1_loss(pred, target)
    # Per-view normalized structure prevents exposure alone from satisfying this term.
    def normalize(x):
        return (x-x.mean((-2,-1),keepdim=True))/x.std((-2,-1),keepdim=True).clamp_min(.05)
    p,t=normalize(pred),normalize(target)
    edges=F.l1_loss(p[:,:,:,1:]-p[:,:,:,:-1],t[:,:,:,1:]-t[:,:,:,:-1])+F.l1_loss(p[:,:,1:,:]-p[:,:,:-1,:],t[:,:,1:,:]-t[:,:,:-1,:])
    return base + .15*edges

def predict(item):
    with torch.autocast('cuda', enabled=False):
        raw = head([[x.cuda()] for x in item['features']], *item['shape'], patch_size=pipeline.config.patch_size)
        return F.elu(raw, alpha=1e-3).float()
def evaluate(split, prefix=None):
    head.eval()
    measurements = []
    with torch.no_grad():
        for item in cache:
            record = item['record']
            if record['split'] != split:
                continue
            pred_log = predict(item).cpu()
            mse_log = F.mse_loss(pred_log, item['target']).item()
            structural_objective=detail_loss(pred_log,item['target']).item()
            pred = (10. ** pred_log - 1.).permute(0, 2, 3, 1).numpy()
            target = (10. ** item['target'] - 1.).permute(0, 2, 3, 1).numpy()
            # Fixed evaluation transform, [0,1] range; HDR is retained separately.
            p = np.maximum(pred, 0.) / (1. + np.maximum(pred, 0.))
            q = np.maximum(target, 0.) / (1. + np.maximum(target, 0.))
            mse = float(np.mean((p - q)**2))
            ssim = np.mean([structural_similarity(a, b, channel_axis=-1, data_range=1.) for a, b in zip(p, q)])
            measurements.append(dict(name=record['name'], split=split, log_mse=mse_log, structural_objective=structural_objective,
                reinhard_psnr=-10 * math.log10(max(mse, 1e-12)), reinhard_ssim=float(ssim)))
            if prefix:
                folder = ROOT / 'results/detail256' / prefix / record['name']
                folder.mkdir(parents=True, exist_ok=True)
                np.save(folder / 'prediction_linear_hdr.npy', pred)
                for view, image in enumerate(p):
                    srgb = np.where(image <= .0031308, image * 12.92, 1.055 * image**(1/2.4) - .055)
                    iio.imwrite(folder / f'view_{view}.png', (np.clip(srgb, 0, 1)*255).astype(np.uint8))
    head.train()
    return measurements
baseline = evaluate('validation', 'baseline') + evaluate('test', 'baseline')
(RUN / 'baseline_metrics.json').write_text(json.dumps(baseline, indent=2))
best = float('inf')
history = []
training = [x for x in cache if x['record']['split'] == 'train']
for epoch in range(1, 101):
    losses = []
    for idx in np.random.permutation(len(training)):
        item = training[idx]
        optimizer.zero_grad(set_to_none=True)
        pred = predict(item)
        loss = detail_loss(pred, item['target'].cuda())
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        if not torch.isfinite(grad) or grad <= 0:
            raise RuntimeError(f'Invalid decoder gradients: {grad}')
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    validation = evaluate('validation')
    val = float(np.mean([x['structural_objective'] for x in validation]))
    entry = dict(epoch=epoch, train_objective=float(np.mean(losses)), validation_objective=val)
    history.append(entry)
    print(json.dumps(entry), flush=True)
    (RUN / 'history.json').write_text(json.dumps(history, indent=2))
    if val < best:
        best = val
        save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, str(RUN / 'best_decoder.safetensors'))
        (RUN / 'best_epoch.json').write_text(json.dumps(entry, indent=2))
from safetensors.torch import load_file
head.load_state_dict(load_file(str(RUN / 'best_decoder.safetensors')))
final = evaluate('validation', 'adapted') + evaluate('test', 'adapted')
(RUN / 'final_metrics.json').write_text(json.dumps(final, indent=2))
(RUN / 'method.json').write_text(json.dumps(dict(method='FP32 DPT decoder-only adaptation; frozen FP16-forward Transformer features',
    epochs=100, learning_rate=1e-5, seed=20261101, resolution=resolution, samples_per_reference=512,
    checkpoint_selection='lowest validation log-L1 plus normalized spatial-gradient objective',
    loss='log HDR L1 plus 0.15 normalized spatial-gradient L1', evaluation_transform='Reinhard x/(1+x), data_range=1, no sRGB for metrics',
    limitations=manifest['sponza_limitations']), indent=2))
print('COMPLETE', json.dumps(final), flush=True)
