"""Embed static evidence panels and copy the reviewed I/O data for GitHub Pages."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
docs = root / 'docs'
(docs / 'io-guide.json').write_bytes((root / 'public-data/io-guide.json').read_bytes())
target = docs / 'index.html'
text = target.read_text(encoding='utf-8-sig')
for name, marker in [('io', '<section id="experiments"'), ('training', '<!-- READING START -->'), ('multiview', '<section id="response-curves"'), ('historical', '<!-- READING START -->')]:
    fragment = (docs / f'{name}-panel.html').read_text(encoding='utf-8')
    start, end = f'<!-- {name.upper()} PANEL START -->', f'<!-- {name.upper()} PANEL END -->'
    content = start + '\n' + fragment + '\n' + end
    if start in text:
        text = re.sub(re.escape(start) + '.*?' + re.escape(end), lambda m: content, text, flags=re.S)
    else:
        text = text.replace(marker, content + '\n' + marker)
script = '<script type="module" src="evidence-panels.js"></script>'
if script not in text:
    text = text.replace('</body>', script + '</body>')
style = '<link rel="stylesheet" href="evidence-panels.css">'
if style not in text:
    text = text.replace('</head>', style + '</head>')
for tag, marker in [('<script type="module" src="multiview.js"></script>', '</body>'), ('<link rel="stylesheet" href="multiview.css">', '</head>')]:
    if tag not in text:
        text = text.replace(marker, tag + marker)
for tag, marker in [('<script type="module" src="historical-regression.js"></script>', '</body>'), ('<link rel="stylesheet" href="historical-regression.css">', '</head>')]:
    if tag not in text:
        text = text.replace(marker, tag + marker)
target.write_text(text, encoding='utf-8')
print('Built I/O and training evidence panels')
