from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from PIL import Image
from xml.sax.saxutils import escape
import json
import re
root=Path(__file__).parent
equation_keys={v:k for k,v in json.loads((root/'equations.json').read_text(encoding='utf-8')).items()}
import sys
private_mode='--private' in sys.argv
profile=json.loads((root/'private'/'profile.json').read_text(encoding='utf-8-sig')) if private_mode else {}
out_dir=root/'private' if private_mode else root
output_stem='RenderFormer-期末报告' if '--final' in sys.argv else 'RenderFormer-期末报告-工作稿'
def cover_text(value):
    replacements={'班    级：________________':'班    级：'+profile.get('class','________________'),'学生姓名：________________':'学生姓名：'+profile.get('name','________________'),'学    号：________________':'学    号：'+profile.get('student_id','________________'),'完成日期：2026 年 ____ 月 ____ 日':'完成日期：'+profile.get('completion_date','2026 年 ____ 月 ____ 日')}
    return replacements.get(value,value)
pdfmetrics.registerFont(TTFont('Song','C:/Windows/Fonts/simsun.ttc',subfontIndex=0))
pdfmetrics.registerFont(TTFont('Hei','C:/Windows/Fonts/simhei.ttf'))
w,h=A4
p=out_dir/(output_stem+'.pdf')
c=canvas.Canvas(str(p),pagesize=A4)
c.setTitle('Transformer 学习方法及其在三角网格神经渲染中的应用');c.setAuthor('')
# Cover reconstructed from textual evidence, not an exact visual reproduction.
c.setLineWidth(.5);c.rect(18,h-180,25,100)
c.setFont('Song',10)
for i,ch in enumerate('装订处'):c.drawCentredString(30,h-110-i*17,ch)
c.setFont('Hei',24);c.drawCentredString(w/2,h-135,'数据科学期末报告')
c.setFont('Hei',17)
for y,s in [(h-225,'Transformer 学习方法及其在'),(h-254,'三角网格神经渲染中的应用'),(h-291,'以 RenderFormer 为例')]:c.drawCentredString(w/2,y,s)
c.setFont('Song',14)
for i,s in enumerate(['院    系：计算机科学与技术','班    级：________________','学生姓名：________________','学    号：________________','完成日期：2026 年 ____ 月 ____ 日']):c.drawString(150,h-395-i*40,cover_text(s))
c.setFont('Song',15);c.drawCentredString(w/2,105,'大连理工大学计算机科学与技术学院')
c.showPage()
body=ParagraphStyle('body',fontName='Song',fontSize=12,leading=15,firstLineIndent=24,spaceAfter=9,wordWrap='CJK')
heading=ParagraphStyle('heading',fontName='Hei',fontSize=12,leading=15,spaceBefore=6,spaceAfter=7,wordWrap='CJK')
refs=ParagraphStyle('refs',parent=body,firstLineIndent=0,fontSize=10.5,leading=14)
text=(root/'report-draft.md').read_text(encoding='utf-8-sig')
pagebreaks={'二 Transformer 的表示与注意力计算','三 RenderFormer 的两阶段网络','四 数据采集与监督学习过程','五 复现实验设计与数据流程','六 评价指标与可复现记录','七 实验结果与讨论','八 总结'}
y=h-65;page=1;bounds=[];image_height_total=0.0;figure_height_total=0.0
def footer():
 c.setFont('Song',9);c.drawCentredString(w/2,32,str(page))
for line in text.splitlines()[3:]:
 if not line.strip():continue
 if line=='<!--results-table-->':
  rows=json.loads((root/'result-table.json').read_text(encoding='utf-8'))
  table=Table(rows,colWidths=[135,80,80,80,80],repeatRows=1)
  table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Song'),('FONTSIZE',(0,0),(-1,-1),9),('LEADING',(0,0),(-1,-1),12),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EEEEEE')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('ALIGN',(1,0),(-1,-1),'CENTER')]))
  tw,th=table.wrap(w-140,h)
  if y-th<55:footer();c.showPage();page+=1;y=h-65
  table.drawOn(c,70,y-th);y-=th+15;continue
 if line.startswith('!['):
  m=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
  if not m:raise ValueError('Invalid figure marker')
  im=Image.open(root/m.group(2));iw=w-140;ih=iw*im.height/im.width
  if ih>180:iw*=180/ih;ih=180
  figure_height_total+=ih
  image_height_total+=ih
  cap=Paragraph(escape(m.group(1)),refs);cw,ch=cap.wrap(w-140,h)
  if y-ih-ch-20<55:footer();c.showPage();page+=1;y=h-65
  c.drawImage(ImageReader(im),(w-iw)/2,y-ih,width=iw,height=ih,mask='auto');y-=ih+7
  cap.drawOn(c,70,y-ch);y-=ch+12;continue
 if line.startswith('$$'):
  key=equation_keys[line.strip('$')];ip=root/'equations'/(key+'.png');im=Image.open(ip);iw=min(390,im.width/4);ih=iw*im.height/im.width
  image_height_total+=ih
  if y-ih<55:footer();c.showPage();page+=1;y=h-65
  c.drawImage(ImageReader(im),(w-iw)/2,y-ih,width=iw,height=ih,mask='auto');y-=ih+15;continue
 if line.startswith('表 1') and y<310:footer();c.showPage();page+=1;y=h-65
 ishead=line.startswith('## ')
 if ishead:
  s=line[3:]
  if y<180:
   footer();c.showPage();page+=1;y=h-65
  style=heading
 else:s=line;style=refs if line.startswith('[') else body
 if ishead:y-=6
 para=Paragraph(escape(s),style)
 pw,ph=para.wrap(w-140,h)
 if y-ph<55:
  footer();c.showPage();page+=1;y=h-65
 para.drawOn(c,70,y-ph);bounds.append({'page':page+1,'bottom':round(y-ph,1),'text':s[:30]});y-=ph+style.spaceAfter
footer();c.save()
(out_dir/'pdf-qa.json').write_text(json.dumps({'pages':page+1,'body_pages':page,'figure_height_total_pt':figure_height_total,'all_image_height_including_equations_pt':image_height_total,'conservative_image_fraction_of_body_printable_area':image_height_total/(page*(h-120)),'fonts':'embedded SimSun and SimHei','min_text_y':min(x['bottom'] for x in bounds),'source':'report-draft.md','template_cover_visual_match':False,'bounds':bounds},ensure_ascii=False,indent=2),encoding='utf-8')
print('PDF pages',page+1)
