"""Compare independent Cycles seeds/samples without replacing frozen reference."""
from pathlib import Path
import json,os
os.environ.setdefault('IMAGEIO_FREEIMAGE_LIB','F:/RenderFormer-work/imageio/freeimage-3.18.0-win64.dll')
import numpy as np,imageio.v3 as iio
from evaluate import measure,display_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d=Path(__file__).resolve().parents[1]/'experiments/teapot/roughness_015'
a=iio.imread(d/'reference.exr')[...,:3];c=iio.imread(d/'reference_8192.exr')[...,:3]
b=iio.imread(d/'prediction/prediction.exr')[...,:3]
result={'reference_4096_vs_8192':measure(a,c),'prediction_vs_frozen4096':measure(a,b),
        'prediction_vs_check8192':measure(c,b),'A4096_max':float(a.max()),'A8192_max':float(c.max()),
        'B_max':float(b.max()),'max_error_pixel_xy':[248,344],
        'A4096_at_pixel':a[344,248].tolist(),'A8192_at_pixel':c[344,248].tolist(),'B_at_pixel':b[344,248].tolist(),
        'note':'4096 spp seed1 remains the frozen primary reference. 8192 spp seed2 is an independent convergence diagnostic. No denoising or adaptive sampling.'}
(d/'highlight-convergence.json').write_text(json.dumps(result,indent=2))
fig,axes=plt.subplots(1,3,figsize=(9,3.2),layout='constrained')
for ax,img,title in zip(axes,[a,c,b],['A · Cycles 4096 spp, seed 1','A check · Cycles 8192 spp, seed 2','B · RenderFormer']):
    ax.imshow(display_transform(img[325:365,225:275]),interpolation='nearest');ax.set_title(title,fontsize=10);ax.axis('off')
fig.suptitle('Fixed highlight crop · x=[225,275), y=[325,365) · same display transform')
fig.savefig(d/'highlight_convergence.png',dpi=180);plt.close(fig)
print(json.dumps(result,indent=2))
