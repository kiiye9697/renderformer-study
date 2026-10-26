const select = document.querySelector('#history-case');
const status = document.querySelector('#history-status');
const output = document.querySelector('#history-results');
const make = (tag, text, cls) => {const node=document.createElement(tag); if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;};
const safe = path => typeof path==='string' && /^assets\/[a-zA-Z0-9_./-]+\.(png|jpg|jpeg|webp)$/.test(path) && !path.includes('..');
let data, group='roughness', items=[], token=0;
const roles=[['reference','A · Cycles 参考'],['pretrained','B · 预训练'],['finetuned','C · 微调']];
async function render(index){
 const id=++token, item=items[index];output.hidden=true;output.replaceChildren();
 if(!item){status.textContent='当前分类尚无记录。';return;}
 const fields=['512 × 512 历史配置'];if(Number.isFinite(item.roughness))fields.push(`粗糙度 ${item.roughness.toFixed(2)}`);if(Number.isFinite(item.wall_strength))fields.push(`墙色强度 ${item.wall_strength.toFixed(1)}`);
 document.querySelector('#history-parameters').textContent=fields.join(' · ');
 if(item.status!=='complete'||!roles.every(([key])=>safe(item[key]))){status.textContent='该历史配置的微调回归尚未完成；不展示模拟对照，也不替换为 256px 测试图。';return;}
 status.textContent='正在加载已完成的 512px 回归对照…';
 try{
  const images=await Promise.all(roles.map(([key])=>new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=reject;img.src=item[key];})));
  if(id!==token)return;
  if(images.some(img=>img.naturalWidth!==512||img.naturalHeight!==512))throw Error('not 512px');
  const grid=make('div',undefined,'training-triptych');roles.forEach(([key,label],i)=>{const fig=make('figure');images[i].alt=`${item.title} ${label}`;fig.append(images[i],make('figcaption',label));grid.append(fig);});output.append(grid);
  if(item.note)output.append(make('p',item.note,'note'));
  if(item.metrics){const detail=make('details');detail.append(make('summary','本配置指标（不与 256px 留出测试合并）'),make('pre',JSON.stringify(item.metrics,null,2)));output.append(detail);}
  if(item.pretrained_error||item.finetuned_error){const detail=make('details');detail.append(make('summary','绝对误差热图：|B−A| / |C−A|'));const errors=make('div',undefined,'training-error-grid');for(const[key,label]of[['pretrained_error','|B−A|'],['finetuned_error','|C−A|']]){if(!safe(item[key]))continue;const fig=make('figure'),img=make('img');img.src=item[key];img.alt=label;img.loading='lazy';img.addEventListener('error',()=>{img.hidden=true;fig.append(make('p','该热图暂不可用。'));});fig.append(img,make('figcaption',label));errors.append(fig);}detail.append(errors,make('p',item.error_note||'色标以本配置记录为准。','note'));if(item.error_scale!==undefined)detail.append(make('p','色标记录：'+JSON.stringify(item.error_scale),'note'));output.append(detail);}
  output.hidden=false;status.textContent='真实 512px 历史回归 · A / B / C 同配置';
 }catch{if(id!==token)return;status.textContent='无法加载同尺寸的 512px 对照；不混用其他分辨率或配置。';}
}
function filter(){
 items=data.cases.filter(item=>(item.groups||[]).includes(group));
 if(group==='wall')items.sort((a,b)=>(a.wall_strength??0)-(b.wall_strength??0));
 if(group==='roughness')items.sort((a,b)=>(a.roughness??0)-(b.roughness??0));
 select.replaceChildren();items.forEach((item,index)=>{const option=make('option',item.title);option.value=index;select.append(option);});select.disabled=!items.length;
 document.querySelectorAll('[data-history-group]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.historyGroup===group)));render(0);
}
select.addEventListener('change',()=>render(Number(select.value)));
document.querySelectorAll('[data-history-group]').forEach(button=>button.addEventListener('click',()=>{if(!data)return;group=button.dataset.historyGroup;filter();}));
fetch('historical-regression.json').then(response=>{if(!response.ok)throw Error('missing record');return response.json();}).then(record=>{if(!Array.isArray(record.cases))throw Error('missing cases');data=record;document.querySelector('#history-note').textContent=record.note;filter();}).catch(()=>{status.textContent='历史回归清单暂不可用，保留待运行状态。';});
