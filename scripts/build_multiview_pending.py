"""Publish camera metadata only, from the prelocked qualitative-showcase manifest.

Usage: python scripts/build_multiview_pending.py PATH_TO_SHOWCASE_MANIFEST
Never reads any image, training frame, held-out test frame, or model output.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('manifest', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
target = root / 'docs/multiview.json'
current = json.loads(target.read_text(encoding='utf-8-sig'))
if any(frame.get('status') == 'complete' for family in current['families'] for frame in family['frames']):
    raise SystemExit('Refusing to overwrite completed measured results with pending metadata')
source = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
if source.get('locked_before_training') is not True:
    raise SystemExit('Expected a showcase locked before training')
angles = [-36, -24, -12, 0, 12, 24]
sponza_labels = ['前方 · 中', '前方 · 左', '前方 · 右', '后方 · 中', '后方 · 左', '后方 · 右']
for family, original in zip(current['families'], ['metal', 'sponza']):
    frames = [item for item in source['frames'] if item['family'] == original]
    if len(frames) != 6:
        raise SystemExit('Expected exactly six prelocked cameras per family')
    family['id'] = original
    family['frames'] = []
    for index, item in enumerate(frames):
        frame = {'id': item['id'], 'status': 'pending', 'label': f'方位 {angles[index]}°' if original == 'metal' else sponza_labels[index], 'camera': item['camera'], 'config_sha256': item['config_sha256']}
        if original == 'metal':
            frame['angle_deg'] = angles[index]
        family['frames'].append(frame)
    family['reference_settings'] = {key: source.get(key) for key in ['resolution', 'cycles_spp', 'cycles_seed']}
current['provenance'] = {'locked_before_training': True, 'purpose': source['purpose']}
current['note'] = '12 个相机在训练前已锁定。最终留出测试完成后才计算独立定性展示输出；这些视图不用于选择模型，也不是独立泛化测试。目前仅公开相机待运行记录，没有模拟图像。'
target.write_text(json.dumps(current, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('Published 12 pending camera metadata records; no images read')
