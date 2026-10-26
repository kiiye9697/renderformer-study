// Read-only visualization of published I/O and training records.
const $ = selector => document.querySelector(selector);
const element = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
const shape = values => '[' + values.join(', ') + ']';
const display = value => value === null || value === undefined ? '未记录' : String(value);

async function loadIO() {
  const response = await fetch('io-guide.json');
  if (!response.ok) throw new Error('I/O record unavailable');
  const data = await response.json();
  const table = $('#io-shapes tbody');
  const meanings = {
    triangles: '场景坐标中的三角形顶点', vn: '逐顶点法线', texture: '13 通道三角形材质块',
    mask: '有效三角形标记；true 表示有效', c2w: '相机到世界的刚体变换', fov: 'pipeline 边界使用角度（°）'
  };
  data.inputs.forEach(input => {
    const row = element('tr');
    row.append(element('td', input.name), element('td', shape(input.shape), 'shape-cell'), element('td', meanings[input.name] || input.meaning));
    table.append(row);
  });
  for (const [label, record] of [['网络输出', data.model_output], ['pipeline 输出', data.pipeline_output]]) {
    const row = element('tr');
    row.append(element('td', label), element('td', shape(record.shape), 'shape-cell'), element('td', record.space));
    table.append(row);
  }
  $('#io-channel-order').textContent = data.texture_channels.map((name, i) => `${i}: ${name}`).join(' · ');
  $('#io-record-version').textContent = `${data.model} · revision ${data.revision} · 代码 ${data.code_commit}`;
  $('#io-load-status').hidden = true;
}

const svgNode = (tag, attributes, text) => {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  if (text !== undefined) node.textContent = text;
  return node;
};

function drawLoss(history, definition) {
  const valid = history.filter(row => Number.isFinite(row.step));
  const train = valid.filter(row => Number.isFinite(row.train_loss));
  const validation = valid.filter(row => Number.isFinite(row.val_loss));
  if (train.length < 2 && validation.length < 2) {
    $('#training-curve-status').textContent = '尚无足够真实记录绘制曲线。需要至少两个已测点；不以模拟曲线或零值填充。';
    return;
  }
  const points = [...train.map(row => [row.step, row.train_loss]), ...validation.map(row => [row.step, row.val_loss])];
  const xs = points.map(row => row[0]), ys = points.map(row => row[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const low = Math.min(...ys), high = Math.max(...ys);
  const pad = Math.max((high - low) * .12, Math.abs(high) * .04, 1e-6);
  const y0 = low - pad, y1 = high + pad;
  const x = value => 64 + (value - x0) / Math.max(x1 - x0, 1) * 610;
  const y = value => 250 - (value - y0) / (y1 - y0) * 210;
  const svg = svgNode('svg', { viewBox: '0 0 720 300', role: 'img', 'aria-label': '真实训练和验证损失记录，横轴为优化器更新步数' });
  svg.append(svgNode('title', {}, '训练与验证损失：只连接真实记录点'));
  for (let i = 0; i < 5; i++) {
    const value = y0 + (y1 - y0) * i / 4;
    svg.append(svgNode('line', { x1: 64, y1: y(value), x2: 674, y2: y(value), stroke: '#d9e5df' }));
    svg.append(svgNode('text', { x: 55, y: y(value) + 4, 'text-anchor': 'end' }, value.toPrecision(3)));
  }
  for (let i = 0; i < 5; i++) {
    const value = x0 + (x1 - x0) * i / 4;
    svg.append(svgNode('text', { x: x(value), y: 274, 'text-anchor': 'middle' }, Number(value.toFixed(1))));
  }
  svg.append(svgNode('text', { x: 370, y: 296, 'text-anchor': 'middle' }, 'Optimizer step / 参数更新步数'));
  for (const [rows, key, color, name] of [[train, 'train_loss', '#16758a', '训练'], [validation, 'val_loss', '#b65c39', '验证']]) {
    rows.sort((a, b) => a.step - b.step);
    if (rows.length > 1) svg.append(svgNode('polyline', { points: rows.map(row => `${x(row.step)},${y(row[key])}`).join(' '), fill: 'none', stroke: color, 'stroke-width': 2 }));
    rows.forEach(row => {
      const dot = svgNode('circle', { cx: x(row.step), cy: y(row[key]), r: 3, fill: color });
      dot.append(svgNode('title', {}, `${name} step ${row.step}: ${row[key]}`));
      svg.append(dot);
    });
  }
  $('#training-curve').append(svg);
  $('#training-curve-status').textContent = `${definition || '损失定义未记录，不能跨任务比较数值。'} 训练 ${train.length} 个点，验证 ${validation.length} 个点。连线仅帮助阅读，不补造缺失测量。`;
  $('#training-legend').hidden = false;
}

function publicImage(path, label) {
  if (typeof path !== 'string' || !/^assets\/[a-zA-Z0-9_./-]+\.(png|jpg|jpeg|webp)$/.test(path) || path.includes('..')) return null;
  const figure = element('figure');
  const image = element('img');
  image.src = path; image.alt = label; image.loading = 'lazy';
  image.addEventListener('error', () => { image.hidden = true; figure.append(element('p', '图像文件暂不可用；不替换为其他结果。')); });
  figure.append(image, element('figcaption', label));
  return figure;
}

function renderTests(data) {
  // Never display held-out frames before the selected checkpoint's final evaluation.
  if (data.test_evaluated !== true) {
    $('#training-test-status').textContent = '留出测试集尚未做最终评估。这里不展示测试帧用于挑模型；最终 checkpoint 确定后，再并列展示 A Cycles 参考 / B 预训练模型 / C 微调模型。';
    return;
  }
  const cases = Array.isArray(data.test_cases) ? data.test_cases : [];
  if (!cases.length) {
    $('#training-test-status').textContent = '记录声明测试已完成，但尚未附带可公开的同条件 A/B/C 图。';
    return;
  }
  $('#training-test-status').textContent = '以下均为最终 checkpoint 的留出测试记录。每例 A/B/C 使用同场景、同相机与统一显示处理；改善与退步都保留。';
  cases.forEach(record => {
    const article = element('article', undefined, 'training-case');
    article.append(element('h4', record.title || record.id));
    const grid = element('div', undefined, 'training-triptych');
    for (const [key, label] of [['reference', 'A · Cycles 参考'], ['pretrained', 'B · 预训练模型'], ['finetuned', 'C · 微调模型']]) {
      grid.append(publicImage(record[key], label) || element('p', `${label}：没有可公开图像`));
    }
    article.append(grid);
    if (record.note) article.append(element('p', record.note, 'note'));
    if (record.metrics) {
      const details = element('details');
      details.append(element('summary', '该例指标与比较口径'), element('pre', JSON.stringify(record.metrics, null, 2)));
      article.append(details);
    }
    $('#training-test-cases').append(article);
  });
}

async function loadTraining() {
  const response = await fetch('training.json');
  if (!response.ok) throw new Error('training record unavailable');
  const data = await response.json();
  const names = { pending: '待正式运行', running: '正在训练 · 尚非最终结果', complete: '运行完成 · 以评估记录为准', failed: '运行失败 · 保留已有记录' };
  $('#training-status').textContent = names[data.status] || '状态尚未核实';
  $('#training-note').textContent = data.note || '训练记录尚未补充。';
  const config = data.config || {};
  const fields = [
    ['Micro-batch', config.micro_batch], ['梯度累积次数', config.gradient_accumulation],
    ['有效 batch', config.effective_batch], ['Epoch', config.epochs],
    ['Optimizer steps', config.optimizer_steps], ['训练 / 验证 / 测试帧', [config.train_frames, config.val_frames, config.test_frames].map(display).join(' / ')]
  ];
  fields.forEach(([label, value]) => {
    const card = element('div');
    card.append(element('span', label), element('b', display(value)));
    $('#training-config').append(card);
  });
  drawLoss(Array.isArray(data.history) ? data.history : [], data.loss_definition);
  renderTests(data);
  $('#training-record').textContent = JSON.stringify({ status: data.status, config: data.config, checkpoint: data.checkpoint ?? null, selection_rule: data.selection_rule ?? null, test_evaluated: data.test_evaluated === true, provenance: data.provenance ?? null }, null, 2);
}
loadIO().catch(() => { $('#io-load-status').textContent = 'I/O 数据暂未加载；请查看原始输入／输出文档。'; });
loadTraining().catch(() => { $('#training-status').textContent = '训练记录暂未加载'; $('#training-note').textContent = '无法读取发布记录；不据此判断是否完成训练。'; });
