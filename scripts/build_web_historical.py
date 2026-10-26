"""Copy final, reviewed historical regression publication into the Pages dataset.

Usage: python scripts/build_web_historical.py PATH_TO_PUBLICATION_JSON
Requires all eight final cases and 512px PNGs. Never reads training/test splits.
"""
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
source = json.loads(args.summary.read_text(encoding='utf-8-sig'))
definitions = {
    'cbox': ('官方康奈尔盒', ['other'], None, None),
    'metal_baseline': ('金属茶壶 · 粗糙度 0.40 / 墙色强度 1.0', ['roughness', 'wall'], .4, 1),
    'metal_roughness_015': ('金属茶壶 · 粗糙度 0.15', ['roughness'], .15, None),
    'metal_roughness_075': ('金属茶壶 · 粗糙度 0.75', ['roughness'], .75, None),
    'metal_wall_000': ('侧墙颜色强度 0.0', ['wall'], .4, 0),
    'metal_wall_050': ('侧墙颜色强度 0.5', ['wall'], .4, .5),
    'sponza_interior': ('Sponza 建筑骨架 · 内部相机', ['other'], None, None),
    'sponza_exterior': ('Sponza 建筑骨架 · 外部相机诊断', ['other'], None, None),
}
cases = source.get('cases', [])
if len(cases) != 8 or {item['id'] for item in cases} != set(definitions):
    raise SystemExit('Expected exactly the eight final historical regression cases')
copies, clean = [], []
for item in cases:
    if item.get('resolution') != 512:
        raise SystemExit('Historical regression must remain 512px')
    title, groups, roughness, wall = definitions[item['id']]
    record = {key: item[key] for key in ['id', 'family', 'resolution', 'metrics', 'error_scale', 'error_note'] if key in item}
    record.update(title=title, groups=groups, status='complete')
    if roughness is not None: record['roughness'] = roughness
    if wall is not None: record['wall_strength'] = wall
    for key in ['reference', 'pretrained', 'finetuned', 'pretrained_error', 'finetuned_error']:
        value = item.get(key)
        if not value:
            if key in ['reference', 'pretrained', 'finetuned']:
                raise SystemExit(f'Missing final image: {item["id"]}/{key}')
            continue
        relative = Path(value)
        if relative.is_absolute() or '..' in relative.parts or not value.startswith('assets/regression/') or relative.suffix != '.png':
            raise SystemExit('Unexpected publication asset path')
        path = args.summary.parent / relative
        header = path.read_bytes()[:24]
        if header[:8] != b'\x89PNG\r\n\x1a\n' or struct.unpack('>II', header[16:24]) != (512, 512):
            raise SystemExit(f'Expected 512x512 PNG: {value}')
        record[key] = value
        copies.append((path, docs / relative))
    clean.append(record)
for src, dst in copies:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
declines = sum(item['metrics']['finetuned']['psnr_display_db'] < item['metrics']['pretrained']['psnr_display_db'] for item in clean)
output = {'schema_version': 1, 'status': 'complete', 'resolution': 512,
          'note': f'原 8 例的 512px 历史回归保留全部配置，{declines}/8 例显示 PSNR 下降。受限适配有收益与代价，不能据此替代通用预训练模型。本区不属于新 256px 留出测试，不合并平均指标，也不作为新的独立泛化证据。',
          'checkpoint_sha256': source.get('checkpoint_sha256'), 'cases': clean}
(docs / 'historical-regression.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'Copied {len(copies)} verified 512px images for eight historical cases')
