"""CPU-only final evidence audit. Does not rerun inference or modify measured results."""
from pathlib import Path
import hashlib,json,math
import numpy as np,imageio.v3 as iio,os
os.environ.setdefault('IMAGEIO_FREEIMAGE_LIB','F:/RenderFormer-work/imageio/freeimage-3.18.0-win64.dll')
ROOT=Path(__file__).resolve().parents[1]
cases=[('cbox',ROOT/'experiments/cbox','reference_4096.exr')]
cases += [(name,ROOT/'experiments/teapot'/name,'reference.exr') for name in ['baseline','roughness_015','roughness_075','wall_000','wall_050']]
cases += [('sponza_interior',Path('F:/RenderFormer-work/sponza-architecture-experiment'),'reference.exr'),('sponza_exterior',Path('F:/RenderFormer-work/sponza-exterior-diagnostic'),'reference.exr')]
rows=[]
for name,d,ref in cases:
    m=json.loads((d/'results/metrics.json').read_text());t=json.loads((d/'prediction/timing.json').read_text())
    assert all(math.isfinite(m[k]) for k in ['psnr_display_db','ssim_display','mae_linear_hdr','rmse_linear_hdr'])
    a=iio.imread(d/ref)[...,:3];b=iio.imread(d/'prediction/prediction.exr')[...,:3]
    assert a.shape==b.shape==(512,512,3) and np.isfinite(a).all() and np.isfinite(b).all()
    g=json.loads((d/'geometry-validation.json').read_text())
    assert g['max_vertex_abs_error']==0 and g['max_normal_abs_error']<.001
    assert g['max_camera_matrix_abs_error']<1e-5 and g['fov_degrees_abs_error']<1e-4
    hashes={label:hashlib.sha256((d/file).read_bytes()).hexdigest() for label,file in [('reference_exr',ref),('prediction_exr','prediction/prediction.exr'),('h5','scene.h5'),('metrics_json','results/metrics.json'),('timing_json','prediction/timing.json')]}
    rows.append({'id':name,'triangles':t['triangles'],
      'psnr_display_db':m['psnr_display_db'],'ssim_display':m['ssim_display'],
      'mae_linear_hdr':m['mae_linear_hdr'],'rmse_linear_hdr':m['rmse_linear_hdr'],
      'steady_mean_seconds':t['steady_mean_seconds'],'geometry_validation':g,'sha256':hashes})
summary={'status':'passed','cases':rows,'checks':['finite 512x512 linear HDR A/B','finite metrics','independent Blender/H5 vertex, normal, camera and FOV agreement','SHA256 source measurements'],
 'timing_note':'Two warm forwards per round only; cbox two-round variability is retained in timing-rounds.json. Not a stable hardware benchmark.'}
(ROOT/'public-data/validation-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
