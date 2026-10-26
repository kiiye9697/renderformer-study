const q = selector => document.querySelector(selector);
const node = (tag, text, cls) => { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (cls) el.className = cls; return el; };
const roles = [['reference', 'A · Cycles 参考'], ['pretrained', 'B · 预训练'], ['finetuned', 'C · 微调']];
let manifest, familyIndex = 0, frameIndex = 0, timer = null, request = 0, loading = false;
['#mv-prev', '#mv-next', '#mv-frame'].forEach(selector => { q(selector).disabled = true; });
const errorDetails = node('details', undefined, 'mv-errors');
errorDetails.hidden = true;
q('#mv-images').insertAdjacentElement('afterend', errorDetails);
const safePath = path => typeof path === 'string' && /^assets\/[a-zA-Z0-9_./-]+\.(png|jpg|jpeg|webp)$/.test(path) && !path.includes('..');
const complete = frame => frame.status === 'complete' && roles.every(([key]) => safePath(frame[key]));
const family = () => manifest.families[familyIndex];
function showErrors(frame) {
  errorDetails.replaceChildren(); errorDetails.hidden = true; errorDetails.open = false;
  if (!frame.pretrained_error && !frame.finetuned_error) return;
  errorDetails.append(node('summary', '展开绝对误差热图：|B−A| 与 |C−A|'));
  const grid = node('div', undefined, 'mv-error-grid');
  for (const [key, label] of [['pretrained_error', '|B 预训练 − A 参考|'], ['finetuned_error', '|C 微调 − A 参考|']]) {
    if (!safePath(frame[key])) { grid.append(node('p', `${label}：尚无可公开热图`)); continue; }
    const figure = node('figure'); const image = node('img'); image.src = frame[key]; image.alt = label; image.loading = 'lazy';
    image.addEventListener('error', () => { image.hidden = true; figure.append(node('p', '热图文件暂不可用，不替换为其他视角。')); });
    figure.append(image, node('figcaption', label)); grid.append(figure);
  }
  errorDetails.append(grid, node('p', frame.error_note || 'display RGB 逐像素通道平均绝对误差；色标范围以当前帧记录为准。', 'note'));
  if (frame.error_scale !== undefined) errorDetails.append(node('p', '统一色标记录：' + JSON.stringify(frame.error_scale), 'note'));
  errorDetails.hidden = false;
}
function stop() { if (timer) clearInterval(timer); timer = null; q('#mv-play').textContent = '播放已完成帧'; q('#mv-play').setAttribute('aria-pressed', 'false'); }
function metadata(frame) {
  return { id: frame.id, angle_deg: frame.angle_deg ?? null, camera: frame.camera ?? null, config_sha256: frame.config_sha256 ?? null, checkpoint_sha256: manifest.checkpoint_sha256 ?? null, geometry_material_light_fixed: family().geometry_material_light_fixed ?? null, geometry_material_light_sha256: family().geometry_material_light_sha256 ?? null, alignment: frame.alignment ?? null, display_transform: family().display_transform ?? null, reference_settings: family().reference_settings ?? null, metrics: frame.metrics ?? null, note: frame.note ?? family().condition_note };
}
async function showFrame(index) {
  const frames = family().frames || [];
  if (!frames.length) return;
  frameIndex = Math.max(0, Math.min(frames.length - 1, index));
  const frame = frames[frameIndex], id = ++request;
  errorDetails.hidden = true; errorDetails.replaceChildren(); errorDetails.open = false;
  loading = false;
  q('#mv-frame').value = frameIndex;
  q('#mv-frame-value').textContent = `${frameIndex + 1} / ${frames.length}`;
  q('#mv-camera-label').textContent = `${frame.label || frame.id}${Number.isFinite(frame.angle_deg) ? ` · 方位角 ${frame.angle_deg}°` : ''}`;
  q('#mv-metadata').textContent = JSON.stringify(metadata(frame), null, 2);
  q('#mv-images').hidden = true;
  q('#mv-images').replaceChildren();
  q('#mv-pending').hidden = false;
  q('#mv-pending h4').textContent = '这一视角尚未完成真实对照';
  q('#mv-status').textContent = complete(frame) ? '正在加载已测图像' : '待运行 · 无模拟图像';
  if (!complete(frame)) return;
  loading = true;
  try {
    const images = await Promise.all(roles.map(([key]) => new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = () => reject(new Error('missing image')); image.src = frame[key]; })));
    if (id !== request) return;
    loading = false;
    if (images.some(image => image.naturalWidth !== images[0].naturalWidth || image.naturalHeight !== images[0].naturalHeight)) throw new Error('image dimensions mismatch');
    roles.forEach(([key, label], i) => {
      const figure = node('figure', undefined, 'mv-card');
      const button = node('button'); button.type = 'button'; button.setAttribute('aria-label', `放大 ${frame.label || frame.id} ${label}`);
      images[i].alt = `${frame.label || frame.id} ${label}`;
      button.append(images[i]);
      button.addEventListener('click', () => { stop(); q('#mv-large').src = frame[key]; q('#mv-large').alt = images[i].alt; q('#mv-large-caption').textContent = images[i].alt; q('#mv-dialog').showModal(); });
      figure.append(button, node('figcaption', label)); q('#mv-images').append(figure);
    });
    q('#mv-pending').hidden = true; q('#mv-images').hidden = false;
    showErrors(frame);
    q('#mv-status').textContent = '真实已测 · 离散视角';
  } catch {
    if (id !== request) return;
    loading = false;
    stop(); q('#mv-status').textContent = '图像未加载或尺寸未对齐'; q('#mv-pending h4').textContent = '暂时无法核对这一组图像';
  }
}
function selectFamily(index) {
  stop(); familyIndex = index; frameIndex = 0;
  const selected = family(), frames = selected.frames || [];
  ++request; loading = false;
  errorDetails.hidden = true; errorDetails.replaceChildren(); errorDetails.open = false;
  q('#mv-images').replaceChildren(); q('#mv-images').hidden = true; q('#mv-pending').hidden = false;
  q('#mv-title').textContent = selected.title;
  q('#mv-condition').textContent = selected.condition_note || '场景一致性记录尚未补充。';
  q('#mv-frame').max = Math.max(0, frames.length - 1);
  q('#mv-frame').disabled = frames.length < 2;
  q('#mv-prev').disabled = frames.length < 2; q('#mv-next').disabled = frames.length < 2;
  q('#mv-play').disabled = frames.filter(complete).length < 2;
  q('#mv-families').querySelectorAll('button').forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
  const initial = frames.findIndex(frame => frame.id === selected.default_frame_id);
  showFrame(initial >= 0 ? initial : 0);
}
q('#mv-frame').addEventListener('input', event => { stop(); showFrame(Number(event.target.value)); });
q('#mv-prev').addEventListener('click', () => { stop(); showFrame((frameIndex - 1 + family().frames.length) % family().frames.length); });
q('#mv-next').addEventListener('click', () => { stop(); showFrame((frameIndex + 1) % family().frames.length); });
q('#mv-play').addEventListener('click', () => {
  if (timer) { stop(); return; }
  const finished = family().frames.map((frame, i) => complete(frame) ? i : -1).filter(i => i >= 0);
  if (finished.length < 2) return;
  q('#mv-play').textContent = `暂停 · ${finished.length} 个已完成视角`; q('#mv-play').setAttribute('aria-pressed', 'true');
  timer = setInterval(() => { if (loading) return; const current = finished.indexOf(frameIndex); showFrame(finished[(current + 1) % finished.length]); }, 1600);
});
q('#mv-close').addEventListener('click', () => q('#mv-dialog').close());
q('#mv-dialog').addEventListener('click', event => { if (event.target === q('#mv-dialog')) q('#mv-dialog').close(); });
document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
fetch('multiview.json').then(response => { if (!response.ok) throw new Error('missing manifest'); return response.json(); }).then(data => {
  if (!Array.isArray(data.families) || !data.families.length) throw new Error('no families');
  manifest = data; q('#mv-note').textContent = data.note;
  data.families.forEach((item, index) => { const button = node('button', item.title); button.type = 'button'; button.addEventListener('click', () => selectFamily(index)); q('#mv-families').append(button); });
  selectFamily(0);
}).catch(() => { q('#mv-note').textContent = '无法读取多视角记录；暂不展示结果。'; q('#mv-prev').disabled = true; q('#mv-next').disabled = true; q('#mv-frame').disabled = true; });
