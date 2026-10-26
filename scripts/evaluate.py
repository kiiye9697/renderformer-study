"""A=Cycles, B=RenderFormer. Common exposure/Reinhard/sRGB and fixed error limits."""
import os, argparse, json
from pathlib import Path
os.environ.setdefault('IMAGEIO_FREEIMAGE_LIB','F:/RenderFormer-work/imageio/freeimage-3.18.0-win64.dll')
import numpy as np, imageio.v3 as iio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def display_transform(x):
    x = np.maximum(x,0) # fixed exposure EV=0
    x = x/(1+x) # channel-wise Reinhard
    return np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055)

def measure(a,b):
    aa,bb=display_transform(a),display_transform(b)
    psnr=float(peak_signal_noise_ratio(aa,bb,data_range=1))
    return {'psnr_display_db':psnr if np.isfinite(psnr) else None,
      'psnr_exact_match':not np.isfinite(psnr),
      'ssim_display':float(structural_similarity(aa,bb,data_range=1,channel_axis=-1)),
      'mae_linear_hdr':float(np.mean(np.abs(a-b))),
      'rmse_linear_hdr':float(np.sqrt(np.mean((a-b)**2)))}

def evaluate(reference,prediction,output,title):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    a=iio.imread(reference)[...,:3].astype(np.float32)
    b=iio.imread(prediction)[...,:3].astype(np.float32)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    aa,bb=display_transform(a),display_transform(b)
    error=np.mean(np.abs(aa-bb),axis=-1)
    iio.imwrite(out/'A_reference.png',np.round(aa*255).astype(np.uint8))
    iio.imwrite(out/'B_prediction.png',np.round(bb*255).astype(np.uint8))
    plt.imsave(out/'error.png',error,cmap='magma',vmin=0,vmax=.25)
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
    convergence=title.startswith('Cycles convergence')
    axes[0].imshow(aa); axes[0].set_title('Cycles · 4096 spp, seed 1' if convergence else 'A · Cycles reference')
    axes[1].imshow(bb); axes[1].set_title('Cycles · 1024 spp, seed 0' if convergence else 'B · RenderFormer prediction')
    m=axes[2].imshow(error,cmap='magma',vmin=0,vmax=.25)
    axes[2].set_title('Mean absolute display error')
    for ax in axes: ax.axis('off')
    fig.colorbar(m,ax=axes[2],fraction=.046,pad=.02,label='Absolute RGB error [0, 1]')
    fig.suptitle(title)
    fig.savefig(out/'comparison.png',dpi=160); plt.close(fig)
    result=measure(a,b)
    result.update({'reference':str(reference),'prediction':str(prediction),
      'shape':list(a.shape),'display_transform':'max(x,0), EV=0, channel-wise Reinhard x/(1+x), IEC sRGB transfer',
      'psnr_ssim_domain':'float32 display-transformed RGB, data_range=1, full image, no crop',
      'ssim_settings':'skimage defaults: 7x7 uniform window, sample covariance',
      'error_scale':[0,.25], 'error_color_saturation_note':'error >=0.25 uses the maximum colormap color',
      'error_saturated_pixel_fraction':float(np.mean(error>=.25)),
      'negative_prediction_fraction':float(np.mean(b<0))})
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--reference',required=True)
    p.add_argument('--prediction',required=True); p.add_argument('--output',required=True)
    p.add_argument('--title',default='Official Cornell box · 512 × 512')
    a=p.parse_args(); evaluate(a.reference,a.prediction,a.output,a.title)
