"""Fixed ROI radiometric responses, preserving A/B identities across parameters."""
from pathlib import Path
import json,csv,os
os.environ.setdefault('IMAGEIO_FREEIMAGE_LIB','F:/RenderFormer-work/imageio/freeimage-3.18.0-win64.dll')
import numpy as np,imageio.v3 as iio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]/'experiments/teapot'
design=json.loads((ROOT/'design.json').read_text())
rows=[]
for case in design['cases']:
    d=ROOT/case['id']
    row={k:case[k] for k in ['id','roughness','red_wall_strength']}
    metrics=json.loads((d/'results/metrics.json').read_text())
    row['psnr_display_db']=metrics['psnr_display_db'];row['ssim_display']=metrics['ssim_display']
    for label,file in [('A',d/'reference.exr'),('B',d/'prediction/prediction.exr')]:
        image=iio.imread(file)[...,:3]
        for roi in ['teapot','floor']:
            x0,y0,x1,y1=design['roi_'+roi]; p=image[y0:y1,x0:x1]
            rgb=np.mean(p,axis=(0,1)); y=p @ np.array([.2126,.7152,.0722])
            row[label+'_'+roi+'_mean_Y']=float(y.mean())
            row[label+'_'+roi+'_p99_Y']=float(np.quantile(y,.99))
            row[label+'_'+roi+'_red_fraction']=float(rgb[0]/max(float(rgb.sum()),1e-8))
    rows.append(row)
with (ROOT/'parameter_responses.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lookup={r['id']:r for r in rows}
fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
groups=[(['roughness_015','baseline','roughness_075'],'roughness','teapot_p99_Y','Roughness $r$ [dimensionless]','Teapot ROI $P_{99}(Y)$ [linear radiance]'),
 (['wall_000','wall_050','baseline'],'red_wall_strength','teapot_red_fraction','Red-wall strength $s$ [dimensionless]','Teapot ROI $R/(R+G+B)$ [dimensionless]'),
 (['wall_000','wall_050','baseline'],'red_wall_strength','floor_red_fraction','Red-wall strength $s$ [dimensionless]','Floor ROI $R/(R+G+B)$ [dimensionless]')]
for ax,(ids,xkey,ykey,xlabel,ylabel) in zip(axes,groups):
    for label,style,color in [('A','o-','#225ea8'),('B','s--','#d95f0e')]:
        ax.plot([lookup[i][xkey] for i in ids],[lookup[i][label+'_'+ykey] for i in ids],style,color=color,label='A · Cycles' if label=='A' else 'B · RenderFormer')
    ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(ROOT/'parameter_curves.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(8.5,3.6),layout='constrained')
for ax,(ids,xkey,xlabel) in zip(axes,[(['roughness_015','baseline','roughness_075'],'roughness','Roughness $r$ [dimensionless]'),(['wall_000','wall_050','baseline'],'red_wall_strength','Red-wall strength $s$ [dimensionless]')]):
    ax.plot([lookup[i][xkey] for i in ids],[lookup[i]['psnr_display_db'] for i in ids],'o-',color='#225ea8')
    ax.set_xlabel(xlabel);ax.set_ylabel('Full-image display PSNR [dB]');ax.grid(alpha=.2)
fig.savefig(ROOT/'metric_curves.png',dpi=180);plt.close(fig)
(ROOT/'parameter_responses.json').write_text(json.dumps({'roi':{k:v for k,v in design.items() if k.startswith('roi')},'rows':rows,
 'interpretation':'Fixed image ROIs selected before predictions; P99 is a robust upper-tail luminance statistic, not a physical highlight-size estimate. Linear values have model-relative radiance units.'},indent=2))
