"""Fixed local safetensors inference with independent load, first, and warm timings."""
import argparse, json, os, sys, time
from pathlib import Path
os.environ['ATTN_IMPL'] = 'sdpa'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/renderformer'))
import torch, numpy as np, imageio.v3 as iio
from infer import load_single_h5_data
from renderformer import RenderFormerRenderingPipeline
p = argparse.ArgumentParser()
p.add_argument('--h5', required=True)
p.add_argument('--output', required=True)
p.add_argument('--model', default='F:/RenderFormer-work/models/renderformer-v1-base')
p.add_argument('--resolution', type=int, default=512)
a = p.parse_args()
assert torch.cuda.is_available(), 'CUDA is required for this experiment'
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
start = time.perf_counter()
pipe = RenderFormerRenderingPipeline.from_pretrained(a.model)
pipe.to(torch.device('cuda')); torch.cuda.synchronize()
load_seconds = time.perf_counter() - start
start = time.perf_counter()
data = load_single_h5_data(a.h5)
h5_read_seconds = time.perf_counter() - start
start = time.perf_counter()
data = {k: v.unsqueeze(0).cuda() for k,v in data.items()}
data['fov'] = data['fov'].unsqueeze(-1)
torch.cuda.synchronize()
input_to_gpu_seconds = time.perf_counter() - start
times = []
torch.cuda.reset_peak_memory_stats()
for run in range(3):
    # Pipeline transforms emission in-place; every run must receive a fresh texture.
    inputs = dict(data); inputs['texture'] = data['texture'].clone()
    torch.cuda.synchronize(); start = time.perf_counter()
    result = pipe(**inputs, resolution=a.resolution, torch_dtype=torch.float16)
    torch.cuda.synchronize(); times.append(time.perf_counter() - start)
    if run == 0:
        hdr = result[0,0].float().cpu().numpy()
        assert np.isfinite(hdr).all()
        np.save(out/'prediction.npy', hdr)
        iio.imwrite(out/'prediction.exr', hdr)
    del result, inputs
metadata = {
 'code_commit': 'eec6a9b8d592303def45e458dd4ebd09d6572abc',
 'model_id': 'microsoft/renderformer-v1-base',
 'model_revision': '8aa0009563789dbb6c7785a26aac5da2c9b6d117',
 'torch': torch.__version__, 'cuda': torch.version.cuda,
 'gpu': torch.cuda.get_device_name(), 'precision': 'fp16 autocast', 'attention': 'sdpa',
 'resolution': a.resolution, 'triangles': data['triangles'].shape[1],
 'load_to_gpu_seconds': load_seconds, 'inference_seconds': times,
 'h5_read_seconds': h5_read_seconds, 'input_to_gpu_seconds': input_to_gpu_seconds,
 'timing_scope': 'complete pipeline forward, both stages, fresh texture clone before timer; no concurrent Cycles task; no scene representation cache',
 'steady_mean_seconds': sum(times[1:])/2,
 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
 'linear_hdr_min': float(hdr.min()), 'linear_hdr_max': float(hdr.max()),
 'h5': a.h5, 'model_path': a.model,
}
(out/'timing.json').write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata, indent=2))
