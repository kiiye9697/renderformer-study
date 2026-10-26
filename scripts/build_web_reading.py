"""Build the on-page reading from the project's original, reviewed Markdown."""
from pathlib import Path
import html
import re

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'research/RenderFormer-精读记录.md'
target = ROOT / 'docs/index.html'


def inline(text):
    text = html.escape(text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<a href="\2">\1</a>', text)
    text = re.sub(r'\[([^\]]+)\]\(([a-zA-Z0-9_./-]+\.md)\)', r'<a href="https://github.com/kiiye9697/renderformer-study/blob/main/research/\2">\1</a>', text)
    return text


parts, toc, paragraph = [], [], []
in_list = False


def flush():
    if paragraph:
        parts.append('<p>' + inline(' '.join(paragraph)) + '</p>')
        paragraph.clear()


for line in source.read_text(encoding='utf-8-sig').splitlines():
    if line.startswith('# '):
        continue
    if not line.strip():
        flush()
        if in_list:
            parts.append('</ul>')
            in_list = False
    elif line.startswith('## '):
        flush()
        title = line[3:]
        anchor = 'reading-' + str(len(toc) + 1)
        toc.append(f'<a href="#{anchor}">{inline(title)}</a>')
        parts.append(f'<h2 id="{anchor}">{inline(title)}</h2>')
    elif line.startswith('### '):
        flush()
        parts.append('<h3>' + inline(line[4:]) + '</h3>')
    elif line.startswith('- '):
        flush()
        if not in_list:
            parts.append('<ul>')
            in_list = True
        parts.append('<li>' + inline(line[2:]) + '</li>')
    else:
        paragraph.append(line)
flush()
if in_list:
    parts.append('</ul>')
section = '''<!-- READING START -->
<section id="paper-reading" class="section reading-section"><div class="section-label">06 / 完整论文精读</div><div class="section-content"><h2>从输入表示，读到学习与边界</h2><p class="reading-lead">下面是完整原创精读记录，涵盖全文、训练规模、消融、失败与发布版本差异。原论文的方法解释与本机实测证据分开书写；可以沿左侧目录逐段阅读，也可以跳回上方的交互网络与真实实验。</p><div class="reading-layout"><aside class="reading-toc" aria-label="精读目录">''' + ''.join(toc) + '</aside><article class="reading-body">' + '\n'.join(parts) + '</article></div></div></section>\n<!-- READING END -->'
text = target.read_text(encoding='utf-8-sig')
if '<!-- READING START -->' in text:
    text = re.sub(r'<!-- READING START -->.*?<!-- READING END -->', lambda m: section, text, flags=re.S)
else:
    text = text.replace('</main>', section + '</main>')
target.write_text(text, encoding='utf-8')
print('Built', len(toc), 'reading chapters')
