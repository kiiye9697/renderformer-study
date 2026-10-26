"""Import all twelve final camera-only showcase cases, bound to prelocked cameras."""
import argparse
import json
import shutil
import struct
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('summary', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
docs = root / 'docs'
target = docs / 'multiview.json'
locked = json.loads(target.read_text(encoding='utf-8-sig'))
source = json.loads(args.summary.read_text(encoding='utf-8-sig'))
cases = source.get('cases', [])
lookup = {case['id']: case for case in cases}
expected = {frame['id'] for family in locked['families'] for frame in family['frames']}
if len(cases) != 12 or set(lookup) != expected:
    raise SystemExit('Expected exactly the twelve prelocked showcase frames')
hashes = source.get('family_fixed_condition_sha256', {})
copies = []
for family in locked['families']:
    family_id = family['id']
    condition_hash = hashes.get(family_id)
    if not condition_hash or len(family['frames']) != 6:
        raise SystemExit('Missing fixed-condition evidence for six-camera family')
    for frame in family['frames']:
        case = lookup[frame['id']]
        if case.get('family') != family_id or case.get('geometry_material_light_sha256') != condition_hash:
            raise SystemExit('Geometry/material/light identity is not constant within family')
        if case.get('camera') != frame.get('camera') or case.get('config_sha256') != frame.get('config_sha256'):
            raise SystemExit('Final camera/config differs from prelocked showcase')
        for key in ['reference', 'pretrained', 'finetuned', 'pretrained_error', 'finetuned_error']:
            value = case.get(key)
            if not value:
                raise SystemExit(f'Missing measured output {frame["id"]}/{key}')
            relative = Path(value)
            if relative.is_absolute() or '..' in relative.parts or not value.startswith('assets/showcase/') or relative.suffix != '.png':
                raise SystemExit('Unexpected showcase asset path')
            path = args.summary.parent / relative
            with path.open('rb') as stream:
                header = stream.read(24)
            if header[:8] != b'\x89PNG\r\n\x1a\n' or struct.unpack('>II', header[16:24]) != (256, 256):
                raise SystemExit(f'Expected 256x256 showcase PNG: {value}')
            copies.append((path, docs / relative))
            frame[key] = value
        for key in ['alignment', 'metrics', 'error_scale', 'error_note']:
            if key in case: frame[key] = case[key]
        frame['status'] = 'complete'
    family['status'] = 'complete'
    family['default_frame_id'] = 'metal-showcase-03' if family_id == 'metal' else family['frames'][0]['id']
    family['geometry_material_light_fixed'] = True
    family['geometry_material_light_sha256'] = condition_hash
    family['display_transform'] = 'EV0; clamp nonnegative for display; channel-wise Reinhard; sRGB'
    family['condition_note'] = '六帧仅改变预锁相机；几何、变换、材质和光源的联合 SHA256 完全相同。A/B/C 共用同一视角与显示处理。' + (' 前后观察位是离散位置，不构成连续 360° 环绕。' if family_id == 'sponza' else '')
    if family_id == 'metal':
        family['condition_note'] += ' −36° 极侧角的参考图已部分裁切茶壶，按预锁配置保留；默认导航为 0°，全部六帧均可查看。微调在侧视角墙沿仍有新增模糊伪影。'
    else:
        family['condition_note'] += ' 微调六个视角仍呈块状模糊，不能视作恢复清晰建筑。'
for src, dst in copies:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
locked['status'] = 'complete'
locked['note'] = '12 个相机在训练前锁定，最终留出测试结束后才计算这组定性展示。全部视角保留，不用于选择模型，也不是独立泛化测试；播放仅逐张切换真实帧，不生成中间视角。微调结果仍存在模糊和伪影。'
locked['checkpoint_sha256'] = source.get('checkpoint_sha256')
target.write_text(json.dumps(locked, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'Imported twelve fixed-condition cameras with {len(copies)} measured images')
