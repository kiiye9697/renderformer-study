"""Typeset measured charts and comparison figures without altering rendered pixels."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'docs/assets'
OUT=ASSETS/'showcase';OUT.mkdir(exist_ok=True)
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
    'axes.unicode_minus':False,'svg.fonttype':'path','font.size':11})

def panels(files,labels,out,width,height):
    fig,axs=plt.subplots(1,len(files),figsize=(width,height),dpi=160)
    fig.subplots_adjust(left=.008,right=.992,bottom=.015,top=.85,wspace=.025)
    for ax,path,label in zip(axs,files,labels):
        ax.imshow(Image.open(path),interpolation='none');ax.set_axis_off()
        ax.set_title(label,fontsize=20,pad=8,color='#172622')
    fig.savefig(OUT/out,dpi=160,facecolor='white');plt.close(fig)

C=ASSETS/'course-control';T=ASSETS/'course-training'
panels([C/'cornell-reference.png',C/'cornell-pretrained.png',C/'cornell-error.png'],
       ['Cycles 参考','官方模型预测','绝对误差  0–0.25'],'official-comparison.png',12,4.25)
panels([T/'initial/validation_0000-reference.png']+[T/f'step-{s:05d}/validation_0000-prediction.png' for s in [500,6000,10000]],
       ['Cycles 参考','500 步','6,000 步','10,000 步'],'training-stages.png',12,3.25)

steps=[0,500,2000,4000,6000,8000,10000]
metrics=[json.loads((T/('initial' if n==0 else f'step-{n:05d}')/'metrics.json').read_text()) for n in steps]
pre=json.loads((T/'pretrained/metrics.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(11,3.55),dpi=160)
fig.subplots_adjust(left=.07,right=.97,bottom=.22,top=.80,wspace=.27)
for ax,key,title,color in zip(axs,['psnr_mean','ssim_mean'],['PSNR / dB','SSIM'],['#187b68','#315faf']):
    y=[m[key] for m in metrics]
    ax.plot(steps,y,color=color,lw=2.5,marker='o',ms=5,label='从头训练')
    ax.axhline(pre[key],ls=(0,(4,4)),color='#7b858c',lw=1.2,label='官方权重 · 同一验证集')
    ax.set_title(title,loc='left',fontsize=13,fontweight='bold',pad=18)
    ax.set_xlabel('更新步数',labelpad=10);ax.set_xticks([0,2000,4000,6000,8000,10000],['0','2k','4k','6k','8k','10k'])
    ax.grid(axis='y',color='#e2e8e6',lw=.8);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#bcc8c4');ax.tick_params(length=0,pad=7,colors='#53645e')
    ax.margins(x=.08,y=.22)
    ax.annotate(f'{y[-1]:.2f}' if key=='psnr_mean' else f'{y[-1]:.4f}',(steps[-1],y[-1]),xytext=(-3,12),textcoords='offset points',ha='right',fontsize=12,color=color,fontweight='bold')
handles,labels=axs[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.63,1.0),ncol=2,frameon=False,fontsize=10)
fig.savefig(OUT/'validation-curves.svg',facecolor='white',metadata={'Date':None});fig.savefig(OUT/'validation-curves.png',facecolor='white',dpi=160);plt.close(fig)
svg_path=OUT/'validation-curves.svg'
svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
print('Built two comparison figures and validation curves from archived measurements.')
