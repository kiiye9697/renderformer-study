"""Build report-ready figures from published measurements; performs no rendering."""
from pathlib import Path
import json
import csv
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
OUT = DOCS / 'assets/badcases'
OUT.mkdir(exist_ok=True)
training = json.loads((DOCS / 'training.json').read_text(encoding='utf-8'))
experiments = json.loads((DOCS / 'experiments.json').read_text(encoding='utf-8'))['experiments']
regression = json.loads((DOCS / 'historical-regression.json').read_text(encoding='utf-8'))['cases']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'savefig.facecolor':'white'})
sources = []

def panel(filename, items, subtitle):
    fig, axes = plt.subplots(1,len(items),figsize=(10,3.5),layout='constrained')
    for ax,(path,label) in zip(axes,items):
        ax.imshow(Image.open(DOCS / path));ax.set_title(label);ax.axis('off')
        sources.append(path)
    fig.suptitle(subtitle,fontsize=11)
    fig.savefig(OUT/filename,dpi=300);plt.close(fig)

sponza=next(x for x in experiments if x['id']=='sponza-architecture')
panel('01-sponza-failure.png',[(sponza['reference'],'A  Cycles reference'),(sponza['prediction'],'B  Pretrained prediction'),(sponza['error'],'Display RGB MAE [0, 0.25]')],
      'Historical Sponza, 512 px: observed failure; multiple factors vary')
metal=next(x for x in training['test_cases'] if x['id']=='metal-g06-v1')
panel('02-metal-regression.png',[(metal['reference'],'A  Cycles reference'),(metal['pretrained'],'B  Pretrained'),(metal['finetuned'],'C  Adapted decoder')],
      'Held-out metal-g06-v1, 256 px: both PSNR and SSIM decrease')
cbox=next(x for x in regression if x['id']=='cbox')
panel('03-cornell-regression.png',[(cbox['reference'],'A  Cycles reference'),(cbox['pretrained'],'B  Pretrained'),(cbox['finetuned'],'C  Adapted decoder')],
      'Historical Cornell, 512 px: adaptation cost outside the selected training family')
fig,axs=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
history=training['history']
axs[0].plot([x['epoch'] for x in history],[x['train_loss'] for x in history],label='Train (evaluation mode)')
axs[0].plot([x['epoch'] for x in history],[x['val_loss'] for x in history],label='Validation')
best=min(history,key=lambda x:x['val_loss'])
axs[0].axvline(best['epoch'],color='#ac3a39',ls='--',label=f"Selected epoch {best['epoch']}")
axs[0].set(xlabel='Epoch',ylabel='MSE in log10(1 + linear HDR)',title='Validation selects the checkpoint')
axs[0].grid(alpha=.2);axs[0].legend(fontsize=7)
for family,color in [('metal','#b45d30'),('sponza','#246b76')]:
    rows=[x for x in training['test_cases'] if x['family']==family]
    axs[1].scatter([x['metrics']['finetuned']['psnr_display_db']-x['metrics']['pretrained']['psnr_display_db'] for x in rows],
                   [x['metrics']['finetuned']['ssim_display']-x['metrics']['pretrained']['ssim_display'] for x in rows],
                   label=f'{family}: all {len(rows)} frames',color=color,s=30)
axs[1].axhline(0,color='gray',lw=.8);axs[1].axvline(0,color='gray',lw=.8)
axs[1].set(xlabel='PSNR change / dB',ylabel='SSIM change',title='All 16 held-out frames; no exclusions')
axs[1].grid(alpha=.2);axs[1].legend(fontsize=7)
fig.savefig(OUT/'04-learning-and-tradeoffs.png',dpi=300);plt.close(fig)
captions=[
('01-sponza-failure.png','历史 Sponza 失败实例。512×512；A 为 Cycles，B 为预训练模型。显示域误差固定色标 [0,0.25]，超过上限的像素饱和。相机、三角形数和场景族同时变化，仅为论文局限的相关实测，不能单独归因于场景内相机。'),
('02-metal-regression.png','256×256 留出配置 metal-g06-v1，A/B/C 同一参考、相机和显示处理。按既有记录选取的退步个例，并非随机代表样本；统计图包含全部 16 帧。'),
('03-cornell-regression.png','512×512 历史 Cornell 回归：冻结主干、仅适配 DPT 仍可能损害其他配置。该组不与 256 像素留出指标合并。'),
('04-learning-and-tradeoffs.png','真实逐轮训练与验证损失，以及全部 16 帧的微调前后 PSNR 和 SSIM 差值。训练继续下降不能代替验证模型选择；金属 8 帧 SSIM 全部下降。')]
with (OUT/'captions.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['file','caption']);w.writerows(captions)
manifest=dict(source_run='historical published RTX 4060 Laptop run-v1; not the new RTX 2080 Ti server run',
    selection='Diagnostic examples selected after results; all-frame scatter includes all 16 test frames',
    display_transform='EV0, negative clamp, channel-wise Reinhard, IEC sRGB; metrics data_range=1',
    error_scale=[0,0.25],paper='https://renderformer.github.io/pdfs/renderformer-paper.pdf',
    source_images=[dict(path=p,sha256=hashlib.sha256((DOCS/p).read_bytes()).hexdigest()) for p in sources],
    figures=[dict(file=n,caption=c) for n,c in captions])
(OUT/'figure-provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('Built four 300 dpi measured figures and provenance; no new model results claimed.')
