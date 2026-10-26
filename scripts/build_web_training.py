"""Import the completed 256px held-out evaluation; no training or model selection.

Usage: python scripts/build_web_training.py PATH_TO_PUBLICATION_TRAINING_SUMMARY
The publication must already contain all 16 final cases and their PNG assets.
"""
import argparse
import json
import re
import shutil
import struct
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('summary', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
docs = root / 'docs'
source = json.loads(args.summary.read_text(encoding='utf-8-sig'))
if source.get('test_evaluated') is not True:
    raise SystemExit('Final held-out evaluation has not been recorded')
cases = source.get('test_cases', [])
if len(cases) != 16 or len({item['id'] for item in cases}) != 16:
    raise SystemExit('Expected all sixteen unique held-out cases, without cherry-picking')
history = source.get('history', [])
if len(history) < 2 or not all('step' in row for row in history):
    raise SystemExit('Expected measured training history with optimizer steps')
config_keys = ['micro_batch', 'gradient_accumulation', 'effective_batch', 'epochs', 'optimizer_steps', 'train_frames', 'val_frames', 'test_frames', 'learning_rate', 'lr', 'resolution', 'trainable_parameters', 'frozen_parameters', 'max_epochs', 'patience', 'relative_min_delta', 'optimizer', 'weight_decay', 'clip_grad_norm', 'seed', 'frozen_backbone', 'cached_feature_dtypes', 'manifest_sha256', 'source_model_sha256', 'official_code_commit', 'torch_version', 'cuda_runtime', 'gpu', 'script_sha256']
config = {key: source.get('config', {}).get(key) for key in config_keys if key in source.get('config', {})}
config['learning_rate'] = config.get('learning_rate', config.get('lr'))
if not all(item.get('shape') == [256, 256, 3] for item in cases):
    raise SystemExit('Expected 256x256x3 final test shape metadata')
config['resolution'] = 256
if config.get('effective_batch') != config.get('micro_batch', 0) * config.get('gradient_accumulation', 0):
    raise SystemExit('Effective batch does not match micro-batch times accumulation')
copies = []
for item in cases:
    for key in ['reference', 'pretrained', 'finetuned', 'pretrained_error', 'finetuned_error']:
        value = item.get(key)
        if not value:
            raise SystemExit(f'Missing final image: {item["id"]}/{key}')
        relative = Path(value)
        if relative.is_absolute() or '..' in relative.parts or not value.startswith('assets/training/') or relative.suffix != '.png':
            raise SystemExit('Unexpected publication image path')
        path = args.summary.parent / relative
        with path.open('rb') as stream:
            header = stream.read(24)
        if header[:8] != b'\x89PNG\r\n\x1a\n' or struct.unpack('>II', header[16:24]) != (256, 256):
            raise SystemExit(f'Expected a 256x256 held-out PNG: {value}')
        copies.append((path, docs / relative))


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, str):
        return re.sub(r'[A-Za-z]:[\\/][^\s\"<>]+', '[local path omitted]', value)
    return value


curve = source.get('loss_curve')
if curve:
    relative = Path(curve)
    if relative.is_absolute() or '..' in relative.parts or not curve.startswith('assets/training/') or relative.suffix != '.png':
        raise SystemExit('Unexpected loss curve path')
    copies.append((args.summary.parent / relative, docs / relative))
keys = ['note', 'loss_definition', 'history', 'test_cases', 'aggregate', 'checkpoint', 'selection_rule', 'provenance', 'stop_reason', 'convergence_interpretation', 'batch_capacity', 'batch_choice', 'loss_curve']
output = {key: source[key] for key in keys if key in source}
output.update(schema_version=1, status='complete', test_evaluated=True, config=config)
best = source.get('checkpoint', {}).get('best_epoch')
output['note'] = f'仅微调 DPT 输出头，冻结预训练场景与视角 Transformer。实际停止于 {config.get("epochs")} epoch / {config.get("optimizer_steps")} 次参数更新，按验证损失选择 epoch {best}；有效 batch 为 {config.get("effective_batch")}。16 帧留出测试只评估这一最终选定模型。'
aggregate = source.get('aggregate', {})
findings = ['受限适配有收益与代价，不能据此替代通用预训练模型。']
for family, label in [('metal', '金属茶壶'), ('sponza', 'Sponza')]:
    record = aggregate.get(family, {})
    a, b = record.get('pretrained', {}), record.get('finetuned', {})
    if all(key in a and key in b for key in ['psnr_display_db', 'ssim_display']):
        findings.append(f'{label}：显示 PSNR {a["psnr_display_db"]:.4f} → {b["psnr_display_db"]:.4f} dB；SSIM {a["ssim_display"]:.4f} → {b["ssim_display"]:.4f}。不同指标的改善不能相互替代。')
metal = [item for item in cases if item.get('family') == 'metal']
down = sum(item['metrics']['finetuned']['ssim_display'] < item['metrics']['pretrained']['ssim_display'] for item in metal)
findings.append(f'金属茶壶 {down}/{len(metal)} 帧 SSIM 下降，需结合逐帧轮廓、背景与高光对照检查，不能写成全面改善。')
if source.get('checkpoint', {}).get('sha256') == '80a66e827b611771b28acf325a196276e7521d93bf54448bc9c736dbd6c8246e':
    findings.append('本次图像目检：metal-g01-v0 的微调图出现背景竖带及轮廓模糊；Sponza 虽有亮度与大致布局改善，仍重度模糊，不能称为结构恢复。此条为该 checkpoint 的定性观察。')
output['findings'] = findings
output = clean(output)
for src, dst in copies:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
(docs / 'training.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'Imported all {len(cases)} held-out cases, {len(history)} measured history points and {len(copies)} images')
