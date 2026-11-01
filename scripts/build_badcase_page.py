"""Publish measured failure evidence without merging unrelated runs."""
from pathlib import Path
import json
import html
import zipfile

root=Path(__file__).resolve().parents[1];docs=root/'docs'
train=json.loads((docs/'training.json').read_text(encoding='utf-8'))
exps=json.loads((docs/'experiments.json').read_text(encoding='utf-8'))['experiments']
reg=json.loads((docs/'historical-regression.json').read_text(encoding='utf-8'))['cases']
figures=json.loads((docs/'assets/badcases/figure-provenance.json').read_text(encoding='utf-8'))['figures']
metal=next(x for x in train['test_cases'] if x['id']=='metal-g06-v1')
cbox=next(x for x in reg if x['id']=='cbox')
paper='https://renderformer.github.io/pdfs/renderformer-paper.pdf'
rows=[
('场景内相机','图13 / 第9页','已有相关实测','Sponza 内外视角已有 A/B；场景复杂度等也变化，不能将全部误差归因于相机。'),
('场景内光源与超大光源','图11 / 第8页','配置已准备，未运行','提供独立改变光源位置或面积的配置；尚无对应新测量。'),
('超过8个光源','图10 / 第8页','配置已准备，未运行','9光源直接预测，与9次单光源线性HDR相加对比；不能相加PNG。'),
('三角形数量与大三角形','图9、12 / 第7—8页','已有相关实测','已测 Sponza 建筑骨架；没有把三角形数与相机等变量独立控制。'),
('分辨率依赖、复杂阴影','图13、14 / 第9页','待独立实验','已有跨分辨率历史数据不构成同场景分辨率消融；需要固定场景重新测量。'),
('高阶镜面反射','图15 / 第9—10页','待独立实验','原论文观察到更高阶反射丢失；现有茶壶不能替代多次反射的受控测试。'),
('微调后的结构与泛化退步','本项目扩展','已有实测','金属8/8帧SSIM下降；历史512像素回归7/8例PSNR下降。')]
table=''.join(f'<tr><th>{a}</th><td>{b}</td><td>{c}</td><td>{d}</td></tr>' for a,b,c,d in rows)
gallery=''.join(f'<figure><a href="assets/badcases/{x["file"]}"><img loading="lazy" src="assets/badcases/{x["file"]}" alt="{html.escape(x["caption"])}"></a><figcaption><b>图 {i+1}</b>　{html.escape(x["caption"])}</figcaption></figure>' for i,x in enumerate(figures))
def delta(metrics,key):return f"{metrics['pretrained'][key]:.4f} → {metrics['finetuned'][key]:.4f}"
source=dict(schema_version=1,kind='paper_limitations_and_measured_diagnostics',
    source_run='Published historical RTX 4060 Laptop run-v1; independent of current server run',
    paper=paper,coverage=[dict(topic=a,paper_location=b,status=c,interpretation=d) for a,b,c,d in rows],
    selected_case_ids=['sponza-architecture','metal-g06-v1','cbox'],
    selection='post-hoc diagnostic cases; not a new independent test set',
    server_training_status='not_synced_or_verified; no new server convergence claim')
(docs/'badcases.json').write_text(json.dumps(source,ensure_ascii=False,indent=2),encoding='utf-8')
page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>失败实例与实验图册 · RenderFormer</title><style>
:root{{color-scheme:light;--ink:#17332f;--muted:#5c6a66;--accent:#a33d2d}}*{{box-sizing:border-box}}body{{margin:0;background:#f7f7f2;color:var(--ink);font:16px/1.8 system-ui,sans-serif}}main{{max-width:1140px;margin:auto;padding:32px 24px 80px}}nav{{display:flex;gap:24px;flex-wrap:wrap;border-bottom:1px solid #d6ded7;padding-bottom:20px}}a{{color:var(--ink);text-underline-offset:4px}}header{{padding:64px 0 24px}}.eyebrow{{letter-spacing:.16em;font-size:12px;color:var(--accent)}}h1{{font-size:clamp(32px,5vw,58px);line-height:1.2;margin:16px 0 24px}}h2{{margin:56px 0 16px;font-size:26px}}.lead{{max-width:800px;font-size:19px}}.note{{color:var(--muted);max-width:900px}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin:36px 0}}.stats div{{border-top:2px solid var(--ink);padding-top:12px}}.stats strong{{display:block;font-size:30px}}figure{{margin:32px 0 48px}}figure img{{width:100%;height:auto;background:white;border:1px solid #d6ded7}}figcaption{{font-size:14px;color:var(--muted);margin-top:12px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;min-width:720px;width:100%;font-size:14px}}th,td{{text-align:left;vertical-align:top;border-bottom:1px solid #d6ded7;padding:14px 12px}}thead{{background:#e7ede6}}th{{min-width:135px}}.download{{display:inline-block;background:var(--ink);color:white;padding:11px 20px;text-decoration:none;margin:12px 12px 12px 0}}footer{{border-top:1px solid #d6ded7;margin-top:60px;padding-top:22px;font-size:13px;color:var(--muted)}}@media(max-width:650px){{main{{padding:24px 16px}}header{{padding-top:40px}}.stats{{grid-template-columns:1fr}}}}@media print{{nav,.download{{display:none}}body{{background:white}}main{{padding:0}}figure{{break-inside:avoid}}h2{{break-after:avoid}}}}
</style></head><body><main><nav><a href="./">← 学习实验室</a><a href="#paper">论文局限</a><a href="#evidence">实测图册</a><a href="#files">报告素材</a></nav><header><p class="eyebrow">MEASURED LIMITS / 可核查的失败与退步</p><h1>误差在哪里，<br>就把实验做到哪里。</h1><p class="lead">把原论文的能力边界，与本项目真正测到的失败分开记录。改善与退步使用同一参考、显示处理和指标定义。</p><p class="note">本页图像与训练结论来自仓库中已发布的历史 RTX 4060 Laptop 实验；不是新服务器的 Bunny／Sponza 训练结果。服务器新一轮未同步，不报告其收敛或效果。</p></header>
<div class="stats"><div><strong>8 / 8</strong>金属留出帧 SSIM 下降</div><div><strong>7 / 8</strong>历史回归配置 PSNR 下降</div><div><strong>16 帧</strong>散点图保留全部留出测试</div></div>
<section id="paper"><h2>原论文指出了什么</h2><p>依据论文第8—9页及图10—15。下表中的“配置已准备”仅表示可生成实验输入，并不表示已运行。相关实测也不等同于单因素因果验证。<a href="{paper}#page=8">打开论文原文 ↗</a></p><div class="scroll"><table><thead><tr><th>条件</th><th>论文位置</th><th>本项目状态</th><th>证据与边界</th></tr></thead><tbody>{table}</tbody></table></div></section>
<section id="evidence"><h2>真实失败与训练代价</h2><p>Sponza 原始内部视角的显示 PSNR 为 7.7802 dB、SSIM 为 0.3203。这个例子同时含相机位置、场景复杂度和资产差异，保留错误但不独断原因。</p><p>留出金属配置 metal-g06-v1 的 PSNR：{delta(metal['metrics'],'psnr_display_db')} dB；SSIM：{delta(metal['metrics'],'ssim_display')}。历史 Cornell 回归 PSNR：{delta(cbox['metrics'],'psnr_display_db')} dB。两组分辨率不同，不合并平均。</p>{gallery}<p class="note">示例按诊断目的事后选取；它们不是新增测试集。全部16帧的平均 log HDR MSE 虽下降65.84%，仍不能抵消金属结构指标退步。固定误差色标为 [0,0.25]；显示图不单独提亮预测。完整逐帧数据见 <a href="training.json">训练记录</a>、<a href="historical-regression.json">历史回归</a>。</p></section>
<section id="files"><h2>用于报告与复现的最小文件</h2><p>四张300 dpi实测图、逐图中文图注和来源哈希已打包。建议在6—10页主体中只选一张紧凑对照图和一张曲线，其余留在网页，保持图片总篇幅不超过1/6。</p><a class="download" href="assets/badcases/report-figures.zip" download>下载报告图包</a><a href="assets/badcases/captions.csv">中文图注 CSV</a> · <a href="assets/badcases/figure-provenance.json">图像来源与哈希</a> · <a href="badcases.json">实验状态表</a><p class="note">只发布PNG、图注、指标和必要复现脚本。权重、特征缓存、原始EXR/H5、场景资产和服务器信息不随网页上传。</p></section><footer>本页不在浏览器内执行神经推理。未运行条件不使用教学图替代实测图。<a href="https://github.com/kiiye9697/renderformer-study">源代码与报告 ↗</a></footer></main></body></html>'''
(docs/'badcases.html').write_text(page,encoding='utf-8')
with zipfile.ZipFile(docs/'assets/badcases/report-figures.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted((docs/'assets/badcases').iterdir()):
        if p.suffix in ['.png','.csv','.json']:z.write(p,p.name)
index=docs/'index.html';text=index.read_text(encoding='utf-8')
if 'badcases.html' not in text:
    text=text.replace('<a href="#paper-reading">精读</a>','<a href="#paper-reading">精读</a><a href="badcases.html">失败实例</a>')
    text=text.replace('<!-- READING END --></main>','''<!-- READING END --><section id="badcase-evidence" class="section"><div class="section-label">FAILURE STUDY / 实测局限</div><div class="section-content"><h2>失败与退步，也是实验结果</h2><p>论文中的场景内相机、复杂光源和高阶反射有哪些限制？我们已测到的 Sponza 失败与微调代价是什么？专区保留原图、固定色标、逐帧指标与未运行状态。</p><p><a class="button" href="badcases.html">查看失败实例与报告图册 →</a></p><p>历史已发布实验与新服务器未同步结果分开记录；不宣称 Bunny 新训练已完成。</p></div></section></main>''')
    index.write_text(text,encoding='utf-8')
print('Built badcases.html, evidence JSON, figure ZIP and index links.')
