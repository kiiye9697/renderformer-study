"""Build the course-facing site and compact report figures from measured server data."""
from pathlib import Path
import json,html,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

R=Path(__file__).resolve().parents[1];D=R/'docs';A=D/'assets/server';F=R/'report/figures'
if (D/'assets/server256/server-training.json').exists():A=D/'assets/server256'
if (D/'assets/server256full/server-training.json').exists():A=D/'assets/server256full'
T=json.loads((A/'server-training.json').read_text());B=json.loads((A/'server-badcases.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
fig,ax=plt.subplots(figsize=(11,3),layout='constrained');ax.set(xlim=(0,11),ylim=(0,3));ax.axis('off')
nodes=[(.05,1.7,2.1,.75,'Triangle attributes\nN × (13 × 32 × 32)'),(2.65,1.7,2.1,.75,'Scene Transformer\n12 self-attention layers'),(5.25,1.7,2.1,.75,'View Transformer\n6 cross-attention layers'),(8.,1.7,2.1,.75,'DPT decoder\nlog-HDR RGB image'),(5.25,.2,2.1,.75,'Camera → ray bundles\nposition, direction, FOV'),(8.,.2,2.1,.75,'Cycles reference\nlog10(1 + RGB) MSE')]
for x,y,w,h,label in nodes:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05',facecolor='#edf2ef' if x<8 else '#f4e8dd',edgecolor='#53665c'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
for start,end in [((2.2,2.07),(2.6,2.07)),((4.8,2.07),(5.2,2.07)),((7.4,2.07),(7.95,2.07)),((6.3,1),(6.3,1.65)),((9.05,1),(9.05,1.65))]:ax.annotate('',end,start,arrowprops={'arrowstyle':'->','color':'#53665c'})
ax.text(.1,.65,f'Coordinates → spatial RoPE; normals → normal encoder\nRay bundles: {(T["resolution"]//8)**2} per view; scene Transformer frozen\nAdaptation: view Transformer + DPT; {T["resolution"]} × {T["resolution"]} output' if A.name=='server256full' else f'Coordinates → spatial RoPE; normals → normal encoder\nFrozen features: 4 × [3, {(T["resolution"]//8)**2}, 768]\nDPT adaptation; {T["resolution"]} × {T["resolution"]} output',fontsize=8.5,va='center')
fig.savefig(F/'server-architecture.png',dpi=300);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,2.1),layout='constrained');ax.set(xlim=(0,11),ylim=(0,2.1));ax.axis('off')
for x,y,label in [(0,0.7,'Scene JSON'),(2.1,.7,'Shared mesh\nvertices / normals'),(4.4,1.25,'H5 → RenderFormer'),(4.4,.15,'Cycles reference'),(7.,.7,'Linear HDR\nEXR'),(9.,.7,'Metrics\nReport / web')]:
 ax.add_patch(FancyBboxPatch((x+.05,y),1.8,.6,boxstyle='round,pad=0.04',facecolor='#edf2ef',edgecolor='#53665c'));ax.text(x+.95,y+.3,label,ha='center',va='center',fontsize=8)
for start,end in [((1.9,1),(2.1,1)),((4,1),(4.4,1.55)),((4,1),(4.4,.45)),((6.3,1.55),(7,1)),((6.3,.45),(7,1)),((8.85,1),(9,1))]:ax.annotate('',end,start,arrowprops={'arrowstyle':'->','color':'#53665c'})
fig.savefig(F/'server-dataflow.png',dpi=300);plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,2.1),layout='constrained');h=T['history']
ax.plot([x['epoch'] for x in h],[x.get('train_objective',x.get('train_log_mse')) for x in h],label='Online training objective')
ax.plot([x['epoch'] for x in h],[x.get('validation_objective',x.get('validation_log_mse')) for x in h],label='Validation objective')
ax.set(xlabel='Epoch',ylabel='Log L1 + normalized edge loss' if T['resolution']==256 else 'Log-HDR MSE');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(F/'server-learning.png',dpi=300);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(7.2,4.3),layout='constrained')
for row,family in enumerate(['bunny','sponza']):
 c=next(x for x in T['cases'] if x['family']==family and x['view']==1)
 for ax,key,label in zip(axes[row],['reference','baseline','adapted'],['Cycles','Pretrained','Adapted']):
  ax.imshow(Image.open(A/c[key+'_image']));ax.axis('off');ax.set_title(f'{family.capitalize()} · {label}',fontsize=9)
fig.savefig(F/'server-test-comparison.png',dpi=300);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(9,4.3),layout='constrained')
for row,family in enumerate(['bunny','sponza']):
 c=next(x for x in T['cases'] if x['family']==family and x['view']==1)
 box=(.25,.10,.72,.55) if family=='bunny' else (.10,.05,.85,.65)
 for ax,key,label in zip(axes[row],['reference','baseline','adapted'],['Cycles','Pretrained','Adapted']):
  im=Image.open(A/c[key+'_image']);w,h=im.size
  ax.imshow(im.crop(tuple(int(v*s) for v,s in zip(box,[w,h,w,h]))));ax.axis('off');ax.set_title(f'{family.capitalize()} · {label}',fontsize=9)
fig.savefig(F/'server-detail-crops.png',dpi=300);plt.close(fig)
(A/'server-detail-crops.png').write_bytes((F/'server-detail-crops.png').read_bytes())
selected=[x for x in B['cases'] if x['id'] in ['control','camera_inside','light_inside','large_light','nine_lights_direct','nine_lights_sum']]
fig,axes=plt.subplots(len(selected),3,figsize=(7.2,2.05*len(selected)),layout='constrained',squeeze=False)
for row,c in enumerate(selected):
 for ax,key,label in zip(axes[row],['reference','prediction','error'],['Cycles','Prediction','MAE [0, 0.25]']):
  ax.imshow(Image.open(A/c[key]));ax.axis('off');ax.set_title(c['id'].replace('_',' ')+' · '+label,fontsize=8)
fig.savefig(F/'server-badcases.png',dpi=300);plt.close(fig)
camera=next(c for c in B['cases'] if c['id']=='camera_inside')
fig,axes=plt.subplots(1,3,figsize=(7.2,2.3),layout='constrained')
for ax,key,label in zip(axes,['reference','prediction','error'],['Cycles','Prediction','MAE [0, 0.25]']):
 ax.imshow(Image.open(A/camera[key]));ax.axis('off');ax.set_title(label,fontsize=9)
fig.savefig(F/'server-camera-case.png',dpi=300);plt.close(fig)
for name in ['server-architecture.png','server-dataflow.png','server-learning.png','server-test-comparison.png','server-badcases.png']:
 (A/name).write_bytes((F/name).read_bytes())
def triple(c,kind='training'):
 keys=['reference','baseline','adapted'] if kind=='training' else ['reference','prediction','error']
 labels=['Cycles','预训练','微调后'] if kind=='training' else ['Cycles','模型预测','绝对误差']
 return '<div class="triptych">'+''.join(f'<figure><img loading="lazy" src="assets/server/{c[k+"_image"] if kind=="training" else c[k]}" alt="{label}"><figcaption>{label}</figcaption></figure>' for k,label in zip(keys,labels))+'</div>'
trainrows='';trainfigs=''
for family,title in [('bunny','Bunny'),('sponza','Sponza')]:
 cases=[x for x in T['cases'] if x['family']==family]
 avg=lambda stage,key:np.mean([x[stage][key] for x in cases])
 trainrows+=f'<tr><th>{title}（3视图）</th><td>{avg("baseline","psnr_db"):.2f} → {avg("adapted","psnr_db"):.2f}</td><td>{avg("baseline","ssim"):.3f} → {avg("adapted","ssim"):.3f}</td><td>{avg("baseline","log_mse"):.5f} → {avg("adapted","log_mse"):.5f}</td></tr>'
 trainfigs+=f'<h3>{title}</h3><div class="view-tabs" role="group" aria-label="{title}视角">'+''.join(f'<button data-family="{family}" data-view="{x["view"]}" aria-pressed="{str(x["view"]==1).lower()}">视图{x["view"]+1}</button>' for x in cases)+'</div>'
 for c in cases:trainfigs+=f'<div data-family-panel="{family}" data-view-panel="{c["view"]}" {"hidden" if c["view"]!=1 else ""}>{triple(c)}</div>'
names={'control':'外部相机与外部光源','camera_inside':'相机移入场景','light_inside':'光源移入场景','large_light':'增大光源面积','nine_lights_direct':'9光源直接预测','nine_lights_sum':'9次单光源预测相加'}
badrows=''.join(f'<tr><th>{names[c["id"]]}</th><td>{c["metrics"]["psnr_db"]:.2f}</td><td>{c["metrics"]["ssim"]:.3f}</td></tr>' for c in B['cases'])
badfigs=''.join(f'<details {"open" if c["id"]=="camera_inside" else ""}><summary>{names[c["id"]]}</summary>{triple(c,"bad")}</details>' for c in B['cases'])
styles='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:#242c28;background:#fafaf7;font:16px/1.85 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1100px;margin:auto;padding:0 28px 60px}nav{display:flex;gap:24px;flex-wrap:wrap;padding:22px 0;border-bottom:1px solid #d4d9d0}a{color:#275d4b;text-underline-offset:4px}nav a{text-decoration:none}header{padding:52px 0 26px}h1{font-size:clamp(26px,4vw,43px);line-height:1.45;max-width:920px;font-weight:650}h2{font-size:26px;margin:0 0 18px}h3{font-size:19px;margin:24px 0 12px}p{max-width:940px;margin:14px 0}.meta,figcaption{font-size:13px;color:#60685f}.kicker{font-size:13px;letter-spacing:.1em;color:#697362}section{border-top:1px solid #d4d9d0;padding:40px 0}figure{margin:0}img{max-width:100%;height:auto}.wide{width:100%;background:white}.triptych{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0 24px}.triptych img{width:100%;background:#eee}.triptych figcaption{text-align:center}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px;margin:20px 0}th,td{text-align:left;border-bottom:1px solid #d4d9d0;padding:12px 14px;vertical-align:top}thead{background:#edf0e9}th{font-weight:600}summary{cursor:pointer;font-weight:600;padding:14px 0}details{border-bottom:1px solid #d4d9d0}button{font:inherit;font-size:14px;padding:5px 16px;background:transparent;border:1px solid #a9b5a5;border-radius:3px;cursor:pointer;margin-right:8px}button[aria-pressed=true]{background:#315d4b;color:white}a.download{display:inline-block;border:1px solid #315d4b;padding:9px 16px;margin:8px 12px 8px 0;text-decoration:none}.equation{padding:12px 0;font-family:Cambria,serif;font-size:20px}.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}footer{border-top:1px solid #d4d9d0;padding-top:24px;font-size:13px;color:#60685f}[hidden]{display:none!important}@media(max-width:650px){main{padding:0 16px 40px}nav{gap:16px}.triptych{gap:5px}.triptych figcaption{font-size:11px}.two{grid-template-columns:1fr}section{padding:28px 0}header{padding-top:28px}}'''
page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="RenderFormer课程实验：Transformer方法、Bunny与Sponza微调、论文边界条件测试。"><title>RenderFormer 复现实验</title><style>{styles}</style></head><body><main><nav><a href="#method">方法</a><a href="#baseline">预训练复现</a><a href="#training">场景微调</a><a href="#limits">边界实验</a><a href="#report">报告与数据</a></nav><header><p class="kicker">数据科学课程实验 · 2026</p><h1>Transformer 在三角网格神经渲染中的应用</h1><p>以 RenderFormer 为例，复现预训练渲染，比较 Bunny 与 Sponza 的受限微调，并检验相机、光源条件变化对预测的影响。</p><p class="meta">RenderFormer V1 Base · 官方代码 eec6a9b8 · 固定权重 8aa00095</p></header>
<section id="method"><h2>方法与网络结构</h2><p>三角形的材质、法线和空间位置构成场景表示。场景 Transformer 在三角形之间交换信息；视角 Transformer 以相机射线束为查询，读取场景特征；DPT 解码器生成图像。训练使用 Cycles 参考图监督。</p><div class="equation">Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V</div><figure><img class="wide" src="assets/server/server-architecture.png" alt="RenderFormer两阶段Transformer及DPT解码器结构"><figcaption>图1　按官方 base 代码整理的网络结构。发布权重采用13通道、32×32的三角形纹理编码；本次仅更新DPT解码器。</figcaption></figure><p><a href="https://renderformer.github.io/pdfs/renderformer-paper.pdf">论文</a> · <a href="https://github.com/microsoft/renderformer">官方代码</a></p></section>
<section id="baseline"><h2>预训练复现与材质实验</h2><p>先用官方康奈尔盒检查几何、相机和材质转换，再改变茶壶粗糙度与侧墙颜色。下图保留原有512像素实测，用于比较参考与模型的材质响应。</p><div class="triptych"><figure><img src="assets/official-cornell/A_reference.png" alt="Cornell参考"><figcaption>Cycles</figcaption></figure><figure><img src="assets/official-cornell/B_prediction.png" alt="Cornell预测"><figcaption>预训练模型</figcaption></figure><figure><img src="assets/official-cornell/error.png" alt="Cornell误差"><figcaption>误差，色标[0,0.25]</figcaption></figure></div><figure><img class="wide" src="assets/parameter_curves.png" alt="粗糙度与侧墙颜色的实测响应曲线"><figcaption>图2　历史512像素参数实验。详细设置与测量见原始数据表，不与下方128像素微调结果合并。</figcaption></figure></section>
<section id="training"><h2>Bunny 与 Sponza 微调</h2><p>冻结预训练 Transformer，使用18帧训练、6帧验证、6帧测试，微调10,595,203个DPT参数。粗糙度0.15、0.5、0.9用于训练，0.3用于验证，0.7留作测试；每个配置包含3个视图。训练和测试共享几何及相近视角，因此这里评价的是固定场景中的材质插值。</p><p class="meta">128×128 · Cycles 512 spp · AdamW，学习率10⁻⁵ · FP32解码器 · 30轮 · 验证最低损失出现在第{T['best_epoch']['epoch']}轮</p><div class="scroll"><table><thead><tr><th>测试场景</th><th>PSNR / dB，前→后</th><th>SSIM，前→后</th><th>log HDR MSE，前→后</th></tr></thead><tbody>{trainrows}</tbody></table></div>{trainfigs}<figure><img class="wide" src="assets/server/server-learning.png" alt="30轮训练和验证损失"><figcaption>图3　训练曲线。训练值是轮内更新损失均值，验证值为轮末测量。30轮预算结束时验证仍在下降，不能据此判定已达到稳定平台。</figcaption></figure><p>Sponza由262,267面简化为19,235面，纹理转为平均颜色，不包含原始透明与法线贴图。微调改善了这组留出配置的整体误差，但128像素图仍缺少细节，不能据此声称完整Sponza或高分辨率渲染已解决。</p></section>
<section id="limits"><h2>论文边界条件实验</h2><p>以同一康奈尔盒为基准，分别移动相机、移动光源、增大光源面积，并比较9光源直接预测与逐光源预测相加。参考与模型使用相同网格、材质及相机；对应论文图10、11、13。</p><div class="scroll"><table><thead><tr><th>条件</th><th>PSNR / dB</th><th>SSIM</th></tr></thead><tbody>{badrows}</tbody></table></div>{badfigs}<p class="meta">512×512 · Cycles 2048 spp · 预训练权重 · 相同EV0、Reinhard与sRGB处理；误差色标[0,0.25]，黑→红→黄。大光源保持辐亮度不变，总功率随面积变化；此项复现的是尺寸压力条件，不是等功率实验。</p><p>光源分解只在线性HDR中相加，随后统一色调映射。神经模型未被强制满足光传输线性，分解后的改善或退步均按实测保留。</p></section>
<section id="report"><h2>报告与复现资料</h2><a class="download" href="https://github.com/kiiye9697/renderformer-study/raw/main/report/RenderFormer-期末报告.pdf">报告 PDF</a><a class="download" href="https://github.com/kiiye9697/renderformer-study/raw/main/report/RenderFormer-期末报告.docx">报告 Word</a><a class="download" href="assets/server/report-figures.zip">报告图与图注</a><p><a href="assets/server/server-training.json">微调逐视图指标与训练曲线</a> · <a href="assets/server/server-badcases.json">边界实验指标</a> · <a href="https://github.com/kiiye9697/renderformer-study/tree/main/scripts">复现脚本</a></p><details><summary>版本与测量说明</summary><p class="meta">微调和边界实验均使用固定官方代码与预训练版本。PSNR、SSIM在浮点显示RGB上计算，data_range=1；log MSE在log10(1+线性HDR)上计算。历史参数实验与本轮微调的分辨率、数据配置不同，分别报告。模型、原始EXR/H5与源资产保留在实验目录。</p></details></section><footer>数据科学期末报告配套实验 · <a href="https://github.com/kiiye9697/renderformer-study">代码仓库</a></footer></main><script>document.querySelectorAll('[data-family]').forEach(b=>b.addEventListener('click',()=>{{const f=b.dataset.family,v=b.dataset.view;document.querySelectorAll('[data-family="'+f+'"]').forEach(x=>x.setAttribute('aria-pressed',x.dataset.view===v));document.querySelectorAll('[data-family-panel="'+f+'"]').forEach(x=>x.hidden=x.dataset.viewPanel!==v)}}));</script></body></html>'''
page=page.replace('<p><a href="https://renderformer.github.io/pdfs/renderformer-paper.pdf">论文</a>', '<figure><img class="wide" src="assets/server/server-dataflow.png" alt="共享网格经过模型与Cycles两条路径得到HDR并计算指标"><figcaption>实验数据流程：两条渲染路径使用同一份变换后的三角形网格，在线性HDR中保留结果，再生成指标与展示图。</figcaption></figure><p><a href="https://renderformer.github.io/pdfs/renderformer-paper.pdf">论文</a>')
page=page.replace('</head>','<link rel="stylesheet" href="portfolio-theme.css"></head>',1)
page=page.replace('<main><nav>','<main><nav><a class="portfolio-return" href="https://www.kiiye9697.cn/projects/">KIIYE9697 / 作品集</a>',1)
if T['resolution']==256:
 page=page.replace('assets/server/',f'assets/{A.name}/').replace('128×128','256×256').replace('128像素','256像素').replace('30轮','100轮').replace('512 spp','2048 spp').replace('0.7留作测试','0.65留作测试').replace('学习率10⁻⁵','学习率5×10⁻⁶')
 page=page.replace('<p class="meta">256×256','<p>本轮使用对数HDR的L1误差与归一化空间梯度误差共同训练，先按每张图、每通道去均值并归一化，再比较边缘变化。归一化减少整体亮度和对比度差异对结构项的影响。</p><p class="meta">256×256',1)
if A.name=='server256full':
 page=page.replace('alt="100轮训练和验证损失"','alt="联合阶段60轮训练与验证目标"').replace('0.65留作测试','0.65用于迭代留出对照').replace('测试场景</th>','留出配置</th>')
 page=page.replace('<figure><img class="wide" src="assets/server256full/server-learning.png"','<figure><img class="wide" src="assets/server256full/server-detail-crops.png" alt="相同像素区域的局部结构对照"><figcaption>固定像素区域裁剪：Bunny耳部和Sponza梁架。各列采用相同裁剪范围。</figcaption></figure><figure><img class="wide" src="assets/server256full/server-learning.png"',1)
 page=page.replace('冻结预训练 Transformer，使用18帧训练、6帧验证、6帧测试，微调10,595,203个DPT参数。','冻结场景Transformer，先训练DPT，再联合适配视角Transformer与DPT，共81,560,713个可训练参数。使用18帧训练、6帧验证和6帧材质留出对照。')
 page=page.replace('本次仅更新DPT解码器。','本轮适配视角Transformer与DPT，场景Transformer保持冻结。').replace('100轮 · 验证最低','DPT100轮＋联合适配60轮 · 联合阶段验证最低')
 page=page.replace('100轮预算结束时验证仍在下降，不能据此判定已达到稳定平台。','图为联合适配阶段，按验证目标选第59轮；轮末损失存在波动。')
 page=page.replace('联合适配60轮 · 联合阶段验证最低','联合适配60轮（360次逐视图更新） · 联合阶段验证最低')
 page=page.replace('学习率5×10⁻⁶','DPT预适配5×10⁻⁶；联合阶段视角网络10⁻⁵、DPT 2×10⁻⁶')
 edges=[]
 for family,label in [('bunny','Bunny'),('sponza','Sponza')]:
  cs=[c for c in T['cases'] if c['family']==family]
  before=np.mean([c['baseline']['structure_edge_l1'] for c in cs]);after=np.mean([c['adapted']['structure_edge_l1'] for c in cs])
  edges.append(f'{label}：{before:.4f} → {after:.4f}（下降{(1-after/before)*100:.1f}%）')
 page=page.replace('<h3>Bunny</h3>','<p>去均值、归一化后的显示RGB梯度误差，'+ '；'.join(edges)+'。该项单独检查边缘变化，数值越低越好。</p><h3>Bunny</h3>',1)
(D/'index.html').write_text(page,encoding='utf-8')
(D/'badcases.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=./#limits"><title>边界实验</title><a href="./#limits">查看边界实验</a></html>',encoding='utf-8')
(D/'badcases.json').write_bytes((A/'server-badcases.json').read_bytes())
with zipfile.ZipFile(A/'report-figures.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in ['server-architecture.png','server-dataflow.png','server-learning.png','server-test-comparison.png','server-badcases.png']:z.write(F/name,name)
 z.write(F/'server-detail-crops.png','server-detail-crops.png')
 z.writestr('captions.txt',f'网络结构：RenderFormer base，两阶段Transformer与DPT。\n场景对照：{T["resolution"]}像素，同相机、同曝光；参考、原始预训练与适配模型。\n训练曲线：验证目标选择模型；训练为轮内更新均值，验证为轮末测量。\n边界实验：512像素，固定误差色标[0,0.25]。\n局部对照：同一像素区域裁剪，不改变显示变换。\n')
print('Built course site and report figures;',len(B['cases']),'measured boundary conditions.')

# Current entry and result-eligibility policy supersede the retired adaptation page.
import runpy
runpy.run_path(str(R/'scripts/build_learning_entry.py'),run_name='__main__')
