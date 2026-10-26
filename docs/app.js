'use strict';
const $=s=>document.querySelector(s);
const stages=[
['三角形成为 token','768 DIMENSIONS','法线经编码与线性投影，材质的 10 维属性经另一分支投影，两者相加形成三角形表示。三个顶点的位置通过相对空间 RoPE 进入注意力，不能简单理解为把所有坐标拼接进 token。'],
['与视角无关的场景编码','12 LAYERS · 6 HEADS · 16 REGISTERS','三角形与 register token 通过双向自注意力交换场景信息。表示不再只包含三角形自身属性，也能携带其他表面的关系。网络学习的关联不应直接等同于真实光路。'],
['以相机射线读取场景','6 LAYERS · 8 × 8 RAY BUNDLES','每个 8×8 射线束组织成查询 token。每层先对场景表示做交叉注意力，再在射线 token 之间做自注意力。最后四层特征经 DPT 解码为图像。'],
['参考图指导参数更新','LOG-DOMAIN L1 + 0.05 LPIPS','训练时将预测与参考 HDR 图比较，以对数域 L1 和色调映射后的感知损失计算误差，反向传播并用 AdamW 更新参数。推理时固定参数，不需要目标场景参考图。']];
function stage(i){const [title,count,body]=stages[i];$('#stage-info').replaceChildren();for(const [tag,text,cls]of [['p',count,'count'],['h3',title,''],['p',body,'']]){const e=document.createElement(tag);e.textContent=text;e.className=cls;$('#stage-info').append(e)}document.querySelectorAll('[role=tab]').forEach((b,k)=>b.setAttribute('aria-selected',String(k===i)));$('#stage-info').setAttribute('aria-labelledby',`stage-${i}`)}
document.querySelectorAll('[role=tab]').forEach((b,i)=>{b.addEventListener('click',()=>stage(i));b.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();const j=(i+(e.key==='ArrowRight'?1:3))%4;stage(j);$(`#stage-${j}`).focus()}})});stage(0);
function attention(){const q=Number($('#query').value);$('#query-value').value=q.toFixed(1);const raw=[-q,0,q].map(Math.exp),sum=raw.reduce((a,b)=>a+b);$('#weights').replaceChildren();raw.forEach((v,i)=>{const row=document.createElement('div');row.className='weight';const title=document.createElement('label');title.textContent=['表面 token A','表面 token B','表面 token C'][i];const pct=document.createElement('span');pct.textContent=(v/sum*100).toFixed(1)+'%';title.append(pct);const track=document.createElement('div'),bar=document.createElement('i');bar.style.width=v/sum*100+'%';track.append(bar);row.append(title,track);$('#weights').append(row)})}$('#query').addEventListener('input',attention);attention();
let experiments=[]; let requestId=0;
function validPath(value){if(typeof value!=='string'||!value||value.startsWith('/')||value.includes('..')||/^[a-z]+:/i.test(value))return null;return value}
function loadImage(path) {
  return new Promise((resolve, reject) => {
    const img = new Image(); img.onload = () => resolve(img); img.onerror = reject; img.src = path;
  });
}
function numericMetrics(x) {
  const m=x.metrics||x.metadata?.metrics||x.metadata||{};
  return Object.entries(m).filter(([k,v])=>/psnr|ssim|mae|rmse/i.test(k)&&typeof v==='number'&&Number.isFinite(v));
}
function metricCards(x) {
  $('#metric-cards').replaceChildren();
  numericMetrics(x).filter(([k])=>!/mae/i.test(k)).slice(0,3).forEach(([k,v])=>{const box=document.createElement('div');box.className='metric-card';const label=document.createElement('span');label.textContent=k.replaceAll('_',' ').toUpperCase();const value=document.createElement('b');value.textContent=v.toFixed(/ssim|rmse/i.test(k)?4:2);box.append(label,value);$('#metric-cards').append(box)});
}
function gallery() {
  $('#result-gallery').replaceChildren();
  experiments.forEach((x,i)=>{const path=validPath(x.prediction);if(x.status!=='complete'||!path)return;const card=document.createElement('button');card.className='result-card';card.type='button';card.dataset.index=i;card.setAttribute('aria-pressed','false');const img=document.createElement('img');img.src=path;img.alt=x.title+' 模型预测';img.loading='lazy';const label=document.createElement('span');label.textContent=x.title;card.append(img,label);card.addEventListener('click',()=>{$('#experiment').value=i;showExperiment()});$('#result-gallery').append(card)});
}
let heroReady=false;
async function showExperiment() {
  const current=++requestId, x=experiments[Number($('#experiment').value)];
  $('#compare-panel').hidden=true; $('#empty-result').hidden=false; $('#error-panel').hidden=true;
  document.querySelectorAll('.result-card').forEach(c=>c.setAttribute('aria-pressed',String(c.dataset.index===$('#experiment').value)));
  if(!x)return;
  $('#scene-credit').hidden=!x.metadata?.source_credit;$('#scene-credit').textContent=x.metadata?.source_credit||'';
  $('#experiment-status').textContent=`${x.status==='complete'?'已运行':x.status==='failed'?'运行失败':'待运行'} · ${x.note||'实验记录待补充'}`;
  const a=validPath(x.reference),b=validPath(x.prediction),error=validPath(x.error);
  if(x.status!=='complete'||!a||!b)return;
  try {
    // Load into independent Image objects: fast selection cannot overwrite earlier onload callbacks.
    const [ref,pred]=await Promise.all([loadImage(a),loadImage(b)]);
    if(current!==requestId)return;
    if(ref.naturalWidth!==pred.naturalWidth||ref.naturalHeight!==pred.naturalHeight)throw Error('尺寸不一致');
    $('#reference').src=a;$('#prediction').src=b;
    $('#compare').style.aspectRatio=`${ref.naturalWidth}/${ref.naturalHeight}`;
    $('#metadata').textContent=JSON.stringify(x.metadata||x.metrics||{},null,2);metricCards(x);
    $('#empty-result').hidden=true;$('#compare-panel').hidden=false;$('#split').value=50;$('#zoom').value=1;view();
    if(!heroReady){heroReady=true;$('#hero-reference').src=a;$('#hero-prediction').src=b;$('#hero-error-box').hidden=!error;$('.hero-triptych').style.gridTemplateColumns=error?'repeat(3,1fr)':'repeat(2,1fr)';if(error)$('#hero-error').src=error;$('#hero-caption').textContent=x.title+' · 本机真实实验。'+(x.note||'');$('#hero-result').hidden=false;$('#hero-pending').hidden=true;}
    if(error){await loadImage(error);if(current!==requestId)return;$('#error-image').src=error;$('#error-panel').hidden=false;if(x.error_note)$('#error-caption').textContent=x.error_note;}
  } catch {
    if(current!==requestId)return;
    $('#experiment-status').textContent='结果文件未能加载或尺寸不一致，请核验实验清单。';
  }
}
function curves(data) {
  const list=Array.isArray(data.curves)?data.curves:[];
  const complete=experiments.filter(x=>x.status==='complete'&&numericMetrics(x).length);
  if(!list.length&&!complete.length)return;
  $('#curve-content').replaceChildren();
  list.forEach(c=>{const path=validPath(c.image||c.path);if(!path)return;const fig=document.createElement('figure');fig.className='curve-figure';const img=document.createElement('img');img.src=path;img.alt=c.title||'参数响应曲线';const caption=document.createElement('figcaption');caption.textContent=c.note||c.title||'真实测量；完整设置见实验记录。';fig.append(img,caption);$('#curve-content').append(fig)});
  if(complete.length){const table=document.createElement('table');table.className='curve-table';const head=document.createElement('tr');['实验配置','指标','实测值'].forEach(t=>{const e=document.createElement('th');e.textContent=t;head.append(e)});table.append(head);complete.forEach(x=>numericMetrics(x).slice(0,2).forEach(([k,v])=>{const row=document.createElement('tr');[x.title,k.toUpperCase(),v.toFixed(/ssim|rmse/i.test(k)?4:2)].forEach(t=>{const cell=document.createElement('td');cell.textContent=t;row.append(cell)});table.append(row)}));$('#curve-content').append(table);if(!list.length){const note=document.createElement('p');note.className='note';note.textContent='当前展示已核验数值；只有同一测量协议的参数组才适合绘制响应曲线。';$('#curve-content').append(note)}}
}
fetch('experiments.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(data=>{experiments=data.experiments||[];$('#experiment').replaceChildren();experiments.forEach((e,i)=>{const o=document.createElement('option');o.value=i;o.textContent=e.title;$('#experiment').append(o)});if(!experiments.length){$('#experiment-status').textContent='没有实验记录。';return}const preferred=experiments.findIndex(e=>e.status==='complete'&&/sponza/i.test(e.id));const first=experiments.findIndex(e=>e.status==='complete');$('#experiment').value=preferred>=0?preferred:first>=0?first:0;gallery();curves(data);showExperiment()}).catch(()=>{$('#experiment').replaceChildren(new Option('实验清单不可用',''));$('#experiment-status').textContent='无法读取实验清单。请通过 HTTP 服务或 GitHub Pages 打开。'});
$('#experiment').addEventListener('change',showExperiment);
function view(){$('#compare').style.setProperty('--split',$('#split').value+'%');$('#compare').style.setProperty('--zoom',$('#zoom').value);$('#split-value').value=$('#split').value+'%';$('#zoom-value').value=$('#zoom').value+'×'}
$('#split').addEventListener('input',view);$('#zoom').addEventListener('input',view);$('#compare').addEventListener('pointermove',e=>{const r=e.currentTarget.getBoundingClientRect();e.currentTarget.style.setProperty('--x',Math.max(0,Math.min(100,(e.clientX-r.left)/r.width*100))+'%');e.currentTarget.style.setProperty('--y',Math.max(0,Math.min(100,(e.clientY-r.top)/r.height*100))+'%')});
const nodes=[
 ['三角网格输入','[B,N,3,3] vertices / normals + material','场景的N个三角形携带几何、顶点法线与材质。相机另行输入；光源以发光三角形表示。原方法参数在每个三角形内常量，不代表支持任意Unity shader。'],
 ['嵌入：物理属性变为可学习特征','paper: material 10 → 768; release base: 13×32×32 → 768; + normal embedding','原论文主方法用10维常量材质，发布base的实测config则采用13×32×32纹理块编码（13312维线性投影）。两者都输出768维；不要混为同一输入结构。法线频率编码、材质投影和可学习triangle token相加，位置经空间RoPE进入注意力。'],
 ['第一阶段：表面之间交换信息','[B,N+16,768] → 12 × (Self-Attention + FFN)','每层6个头，Q/K/V来自同一场景序列；16个register允许存放全局信息。归一化、残差和SwiGLU辅助学习。层数不是光线反弹次数；输出隐表示不能按输入材质通道直接解释。'],
 ['相机定义我们要回答的问题','resolution H×W → H×W normalized ray directions','把世界位置转换到相机坐标后，相机处于原点。每条射线经过一个像素中心，视场角决定方向。该步骤产生查询，不执行光线求交或路径追踪。'],
 ['64条射线组成一个查询','8×8×3 = 192 → Linear + RMSNorm → 768','一个token代表8×8像素块。512²图像对应4096个射线束，256²对应1024个。射线束方向输入提供相机几何信息；没有输入参考图。'],
 ['第二阶段：查询场景并交流上下文','Q: ray tokens; K,V: scene tokens → Cross-Attn → Self-Attn → FFN','6层中每层先跨序列读取场景，然后在射线束间交换信息。空间RoPE在相机坐标下编码三角形位置。一般注意力式只是基式，实际还采用QK-Norm等。'],
 ['解码：从特征到HDR','last 4 view layers → DPT → log HDR → linear HDR','DPT组合最后四层特征得到稠密图。当前发布pipeline使用10**prediction−1恢复线性HDR。可视化时另作统一显示变换；PNG不替代原EXR。']
];
let activeNode=2; let measuredData=null;
function nodeDetail(i){activeNode=i;const [title,shape,body]=nodes[i];const panel=$('#network-detail');panel.replaceChildren();for(const [tag,text,cls]of[['div','TEACHING EXPLANATION · NOT MEASURED ACTIVATION','detail-tag'],['h3',title,''],['code',shape,''],['p',body,'']]){const e=document.createElement(tag);e.textContent=text;e.className=cls;panel.append(e)}appendMeasuredNode(panel,i);panel.hidden=false;$('#network-expand').setAttribute('aria-expanded','true');$('#network-expand').textContent='收起计算细节 −';document.querySelectorAll('#network-buttons button').forEach((b,j)=>b.setAttribute('aria-pressed',String(i===j)));document.querySelectorAll('.diagram-node').forEach(n=>n.classList.toggle('active',Number(n.dataset.node)===i))}
nodes.forEach((x,i)=>{const b=document.createElement('button');b.type='button';b.textContent=String(i+1).padStart(2,'0')+' '+x[0].split('：')[0];b.setAttribute('aria-pressed','false');b.addEventListener('click',()=>nodeDetail(i));$('#network-buttons').append(b)});
document.querySelectorAll('.diagram-node').forEach(n=>n.addEventListener('click',()=>nodeDetail(Number(n.dataset.node))));
$('#network-expand').addEventListener('click',()=>{if($('#network-detail').hidden)nodeDetail(activeNode);else{$('#network-detail').hidden=true;$('#network-expand').setAttribute('aria-expanded','false');$('#network-expand').textContent='展开计算细节 ＋'}});
fetch('shapes.json').then(r=>r.ok?r.json():null).then(data=>{if(data){measuredData=data;$('#actual-shapes').textContent=JSON.stringify(data,null,2);if(!$('#network-detail').hidden)nodeDetail(activeNode)}}).catch(()=>{});

function tensorSummary(v){
  if(Array.isArray(v))return v.flatMap(tensorSummary);
  if(v&&typeof v==='object'&&Array.isArray(v.shape))return [v];
  return [];
}
function appendMeasuredNode(panel,i){
  if(!measuredData||measuredData.status!=='measured')return;
  const names=[[],['vn_encoding_proj','texture_encoder'],['transformer'],['view_transformer'],['view_transformer.ray_map_encoder'],['view_transformer.transformer'],['view_transformer.out_dpt']][i];
  const box=document.createElement('div');box.className='measured-node';const heading=document.createElement('h4');heading.textContent='本机真实 hook · 官方案例 / 512²';box.append(heading);
  if(!names.length){const info=document.createElement('p');info.textContent=measuredData.source_scene+'。几何输入未在本节点单独hook；不把推导形状伪称采样。';box.append(info)}
  names.forEach(name=>{const rec=measuredData.records.find(r=>r.module===name);if(!rec)return;const title=document.createElement('b');title.textContent=name;box.append(title);const inputs=tensorSummary(rec.inputs),outputs=tensorSummary(rec.outputs);const line=document.createElement('code');line.textContent=inputs.map(v=>'['+v.shape.join(',')+']').join(' + ')+' → '+outputs.map(v=>'['+v.shape.join(',')+']').join(' + ');box.append(line);if(outputs[0]){const v=outputs[0],stat=document.createElement('p');stat.textContent=`首个输出：${v.dtype}，mean=${v.mean.toPrecision(4)}，std=${v.std.toPrecision(4)}，range=[${v.min.toPrecision(4)}, ${v.max.toPrecision(4)}]`;box.append(stat)}});
  const note=document.createElement('p');note.className='note';note.textContent='统计来自单独诊断前向，不计入性能测量；不是attention热图，也不是物理照度。';box.append(note);panel.append(box);
}
