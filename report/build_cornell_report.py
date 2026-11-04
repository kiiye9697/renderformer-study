"""Build the local stage report; no remote access, training, test evaluation or publication."""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
import argparse,hashlib, json, math, re, statistics, shutil
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
CORNELL = STUDY / 'experiments/cornell'
OUTPUT = STUDY / 'report/RenderFormer-课程报告.docx'
QA = STUDY / 'report/qa/cornell-stage'
REFERENCE = STUDY / 'report/template-reference.docx'
STEPS = [0, 500, 2000, 4000, 6000, 8000, 10000]

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def table_md(headers, rows):
    return '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in [headers, ['---']*len(headers), *rows])

def content():
    records = []
    expected = {(f'validation_{i:04d}', v) for i in range(32) for v in range(4)}
    for step in STEPS:
        label = 'initial' if step == 0 else f'step-{step:05d}'
        m = read_json(CORNELL / 'evaluations' / label / 'metrics.json')
        assert m['split'] == 'validation' and m['full_validation']
        assert m['configurations'] == 32 and m['images'] == len(m['rows']) == 128
        assert {(r['id'], r['view']) for r in m['rows']} == expected
        assert m['metric_space'] == 'Reinhard linear RGB before sRGB; data_range=1'
        for field in ['mse','psnr','ssim','gradient_l1']:
            values = [r[field] for r in m['rows']]
            assert all(math.isfinite(v) for v in values)
            assert math.isclose(statistics.mean(values),m[field+'_mean'],abs_tol=1e-10)
        records.append(dict(step=step, **m))
    best = min(records,key=lambda m:m['mse_mean'])
    assert best['step'] == 10000, 'Update narrative and figures before changing this snapshot.'
    analysis = read_json(CORNELL/'report/learning-analysis.json')
    assert analysis['seed_count']==1 and analysis['view_index']==0
    diag = {r['step']:r for r in analysis['rows']}
    assert all(diag[s]['views']==4 for s in STEPS)
    assert diag[6000]['residual_l1'] < diag[0]['residual_l1']
    assert all(diag[s]['residual_l1'] >= diag[0]['residual_l1'] for s in [500,2000,4000])
    pretrained=read_json(CORNELL/'evaluations/pretrained/metrics.json')
    assert pretrained['full_validation'] and pretrained['images']==128 and len(pretrained['rows'])==128
    assert {(r['id'],r['view']) for r in pretrained['rows']}==expected
    for field in ['psnr','ssim','mse']:
        assert math.isclose(statistics.mean(r[field] for r in pretrained['rows']),pretrained[field+'_mean'],abs_tol=1e-10)
    official=read_json(STUDY/'site/docs/assets/course-control/cornell-measurements.json')
    validation = table_md(['模型或更新步数','PSNR / dB','SSIM'],[
        ['官方预训练',f"{pretrained['psnr_mean']:.4f}",f"{pretrained['ssim_mean']:.5f}"],
        *[['随机初始化' if m['step']==0 else m['step'],f"{m['psnr_mean']:.4f}",f"{m['ssim_mean']:.5f}"] for m in records if m['step'] in [0,500,6000,10000]]])
    diagnostic = table_md(['更新步数','低通L1','残差L1'],[
        ['初始化' if s==0 else s,f"{diag[s]['lowpass_l1']:.5f}",f"{diag[s]['residual_l1']:.5f}"] for s in [0,500,6000,8000,10000]])
    start = diag[0]['regions']['tall_box_edges']['gradient_l1']
    end = diag[10000]['regions']['tall_box_edges']['gradient_l1']
    bias = diag[10000]['regions']['left_wall']['rgb_signed_bias']
    region = (f'例如，10000步左墙区域的预测减参考RGB均值为（{bias[0]:+.4f}，{bias[1]:+.4f}，{bias[2]:+.4f}），反映三个通道偏差方向。'
              f'高立方体区域梯度L1从初始化的{start:.5f}变为{end:.5f}，说明整体指标改善并未保证该区域边缘误差下降。')
    replacements = dict(validation_table=validation,diagnostic_table=diagnostic,
        official_psnr=f"{official['psnr']:.2f}",official_ssim=f"{official['ssim']:.4f}",pretrained_psnr=f"{pretrained['psnr_mean']:.2f}",
        best_psnr=f"{best['psnr_mean']:.5f}",best_ssim=f"{best['ssim_mean']:.6f}",best_mse=f"{best['mse_mean']:.8f}",
        initial_gradient=f"{records[0]['gradient_l1_mean']:.5f}",best_gradient=f"{best['gradient_l1_mean']:.5f}",
        lowpass_drop=f"{100*(1-diag[500]['lowpass_l1']/diag[0]['lowpass_l1']):.1f}",
        residual_rise=f"{100*(diag[500]['residual_l1']/diag[0]['residual_l1']-1):.1f}",regional_observation=region)
    text = (HERE/'report-content.md').read_text(encoding='utf-8')
    for key,value in replacements.items():
        text = text.replace('{{'+key+'}}',value)
    assert '{{' not in text
    return text,records

def disable_grid(p):
    el=OxmlElement('w:snapToGrid'); el.set(qn('w:val'),'0'); p._p.get_or_add_pPr().append(el)

def math_run(text):
    r=OxmlElement('m:r');t=OxmlElement('m:t');t.text=text;r.append(t);return r

def math_box(tag, children):
    e=OxmlElement('m:'+tag)
    for child in children:e.append(math_run(child) if isinstance(child,str) else child)
    return e

def sub(base,index):
    return math_box('sSub',[math_box('e',[base]),math_box('sub',[index])])

def sup(base,index):
    return math_box('sSup',[math_box('e',[base]),math_box('sup',[index])])

def hat(base):
    pr=OxmlElement('m:accPr');c=OxmlElement('m:chr');c.set(qn('m:val'),'̂');pr.append(c)
    return math_box('acc',[pr,math_box('e',[base])])

def fraction(numerator,denominator):
    return math_box('f',[math_box('num',numerator),math_box('den',denominator)])

def equation(text):
    if text.startswith('Q ='):
        children=['Q = X',sub('W','Q'),',  K = X',sub('W','K'),',  V = X',sub('W','V')]
    elif text.startswith('Attention('):
        pr=OxmlElement('m:radPr');hide=OxmlElement('m:degHide');hide.set(qn('m:val'),'1');pr.append(hide)
        root=math_box('rad',[pr,math_box('deg',[]),math_box('e',[sub('d','k')])])
        children=['Attention(Q,K,V) = softmax(',fraction(['Q',sup('K','T')],[root]),')V']
    elif text.startswith('Z ='):
        children=['Z = ',sub('log','10'),'(1 + I),  L = mean|',hat('Z'),' − Z| + 0.05 ',sub('L','LPIPS')]
    elif text.startswith('Î ='):
        children=[hat('I'),' = ',sup('10',hat('Z')),' − 1,  R(x) = ',fraction(['max(x,0)'],['1 + max(x,0)'])]
    else:children=[text]
    return math_box('oMath',children)

def build(text):
    inventory=read_json(QA/'template-inventory.json')
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest()==inventory['sha256']
    # Start each build from a fresh physical copy of the teacher's document.
    working_copy=QA/'template-working-blank-cover.docx'
    shutil.copyfile(REFERENCE,working_copy)
    assert working_copy.read_bytes()==REFERENCE.read_bytes()
    d=Document(working_copy)
    for p in list(d.paragraphs)[18:]:
        p._element.getparent().remove(p._element)
    # The complete original cover is untouched; the user will fill it by hand.
    # Hide the first-page footer and number the six body pages from 1.
    section=d.sections[0]
    section.different_first_page_header_footer=True
    # Explicit empty headers prevent Word from applying the template's latent
    # Header style (which includes a rule) when first-page layout is enabled.
    for header in [section.header,section.first_page_header]:
        header.is_linked_to_previous=False
        header.paragraphs[0].style=d.styles['Normal']
        pr=header.paragraphs[0]._p.get_or_add_pPr()
        pr.get_or_add_pStyle().val=d.styles['Normal'].style_id
        borders=OxmlElement('w:pBdr')
        for edge in ['top','left','bottom','right','between','bar']:
            border=OxmlElement('w:'+edge);border.set(qn('w:val'),'nil');borders.append(border)
        pr.append(borders)
    section.first_page_footer.is_linked_to_previous=False
    section.footer.is_linked_to_previous=False
    page_number=OxmlElement('w:pgNumType');page_number.set(qn('w:start'),'0')
    section._sectPr.insert_element_before(page_number,'w:cols','w:titlePg','w:docGrid')
    footer=section.footer.paragraphs[0]
    footer.alignment=1;footer.paragraph_format.first_line_indent=Pt(0)
    footer.paragraph_format.space_before=Pt(0);footer.paragraph_format.space_after=Pt(0)
    footer.paragraph_format.line_spacing=1;disable_grid(footer)
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),' PAGE ')
    run=OxmlElement('w:r');properties=OxmlElement('w:rPr');fonts=OxmlElement('w:rFonts')
    for script in ['ascii','hAnsi','eastAsia','cs']:fonts.set(qn('w:'+script),'宋体')
    properties.append(fonts);size=OxmlElement('w:sz');size.set(qn('w:val'),'18');properties.append(size)
    run.append(properties);value=OxmlElement('w:t');value.text='1';run.append(value)
    field.append(run);footer._p.append(field)
    # Separate styles preserve inherited cover typography.
    for name,font,size in [('CourseBody','宋体',12),('CourseHeading','黑体',12),('CourseSubheading','黑体',12),('CourseCaption','宋体',9),('CourseReference','宋体',9)]:
        st=d.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH);st.base_style=d.styles['Normal']
        st.font.name=font;st.font.size=Pt(size);st.font.color.rgb=RGBColor(0,0,0);st.font.bold=False
        for script in ['ascii','hAnsi','eastAsia','cs']:
            st._element.get_or_add_rPr().rFonts.set(qn('w:'+script),font)
        f=st.paragraph_format;f.line_spacing=1;f.space_after=Pt(0);f.space_before=Pt(0)
        f.first_line_indent=Cm(.8467) if name=='CourseBody' else Cm(0);f.widow_control=True
        if name=='CourseBody':f.alignment=3
        if name in ['CourseHeading','CourseSubheading']:
            f.keep_with_next=True;f.page_break_before=name=='CourseHeading'
            st._element.get_or_add_pPr().find(qn('w:spacing')).set(qn('w:beforeLines'),'50')
        if name=='CourseCaption':f.alignment=1
        if name=='CourseReference':f.alignment=0
    def para(value,style='CourseBody'):
        p=d.add_paragraph(value,style);disable_grid(p);return p
    def table(lines):
        rows=[[v.strip() for v in ln.strip().strip('|').split('|')] for ln in lines]
        rows=[row for row in rows if not all(re.fullmatch(r':?-+:?',v) for v in row)]
        t=d.add_table(rows=0,cols=len(rows[0]));t.autofit=False
        width=d.sections[0].page_width-d.sections[0].left_margin-d.sections[0].right_margin
        ratios=([.22,.78] if len(rows[0])==2 else [.50,.25,.25] if rows[0][1]=='PSNR / dB' else [.25,.43,.32])
        for col,ratio in zip(t.columns,ratios):col.width=int(width*ratio)
        for ri,values in enumerate(rows):
            cells=t.add_row().cells;trpr=cells[0]._tc.getparent().get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
            if ri==0:trpr.append(OxmlElement('w:tblHeader'))
            for ci,(c,value) in enumerate(zip(cells,values)):
                c.width=int(width*ratios[ci])
                c.text=value;c.vertical_alignment=1;pr=c._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
                for edge in ['top','left','bottom','right']:
                    e=OxmlElement('w:'+edge)
                    for key,val in [('val','single'),('sz','4'),('color','D9D9D9')]:e.set(qn('w:'+key),val)
                    b.append(e)
                pr.append(b)
                if ri==0:
                    fill=OxmlElement('w:shd');fill.set(qn('w:fill'),'EEEEEE');pr.append(fill)
                for p in c.paragraphs:
                    p.style=d.styles['CourseBody'];p.paragraph_format.first_line_indent=Cm(0)
                    p.paragraph_format.alignment=1 if rows[0][1]=='PSNR / dB' and ci>0 else 0
                    p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(3);disable_grid(p)
                    for run in p.runs:run.font.size=Pt(10)
        spacer=para('');spacer.paragraph_format.space_after=Pt(0);spacer.paragraph_format.line_spacing=Pt(5)
    figures=[];lines=text.splitlines();idx=0;refs=False
    while idx<len(lines):
        line=lines[idx].strip();idx+=1
        if not line or line.startswith('# '):continue
        if line.startswith('## '):para(line[3:],'CourseHeading');continue
        if line.startswith('### '):
            para(line[4:],'CourseSubheading');refs=line[4:]=='参考文献';continue
        if line.startswith('|'):
            group=[line]
            while idx<len(lines) and lines[idx].strip().startswith('|'):
                group.append(lines[idx]);idx+=1
            table(group);continue
        if line.startswith('$$ '):
            p=para('');p.alignment=1;p.paragraph_format.first_line_indent=Cm(0)
            p._p.append(equation(line[3:-3]));continue
        match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
        if match:
            caption,filename=match.groups();path=(HERE/filename).resolve()
            p=para('');p.alignment=1;p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.keep_with_next=True
            shape=p.add_run().add_picture(str(path),width=Cm(13 if 'official-comparison' in filename else 14))
            shape._inline.docPr.set('descr',caption)
            figures.append(dict(path=str(path.relative_to(STUDY)),width_cm=shape.width.cm,height_cm=shape.height.cm,area_cm2=shape.width.cm*shape.height.cm))
            para(caption,'CourseCaption');continue
        p=para(line,'CourseReference' if refs else 'CourseBody')
        if line.startswith('对Sponza的检查'):p.paragraph_format.page_break_before=True
        if '完整commit见文献' in line:p.paragraph_format.alignment=0
        if re.match(r'^表\d+\s',line):p.style=d.styles['CourseCaption'];p.paragraph_format.keep_with_next=True
    d.core_properties.title=text.splitlines()[0][2:]
    d.core_properties.subject='Transformer方法与RenderFormer实验；数据快照2026-11-03'
    d.core_properties.author='';d.core_properties.comments=''
    generated=BytesIO();d.save(generated)
    allowed={n for n,e in inventory['parts'].items() if e['policy']=='editable'}
    with ZipFile(REFERENCE) as original,ZipFile(generated) as edited,ZipFile(OUTPUT,'w',ZIP_DEFLATED) as out:
        for name in original.namelist():out.writestr(name,edited.read(name) if name in allowed else original.read(name))
        for name in edited.namelist():
            if name not in original.namelist():out.writestr(name,edited.read(name))
    with ZipFile(REFERENCE) as original,ZipFile(OUTPUT) as final:
        for name,entry in inventory['parts'].items():
            if entry['policy']=='preserve':assert final.read(name)==original.read(name),name
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        a,b=[etree.fromstring(z.read('word/document.xml')) for z in [original,final]]
        original_section=a.find('.//w:sectPr',ns)
        final_section=b.find('.//w:sectPr',ns)
        # Only page-number/footer controls may be added to the original section.
        for tag in ['headerReference','footerReference','titlePg','pgNumType']:
            for node in final_section.findall('w:'+tag,ns):final_section.remove(node)
        assert etree.tostring(original_section)==etree.tostring(final_section)
        original_cover=a.findall('./w:body/w:p',ns)[:18]
        final_cover=b.findall('./w:body/w:p',ns)[:18]
        for i,(pa,pb) in enumerate(zip(original_cover,final_cover)):
            assert etree.tostring(pa)==etree.tostring(pb),i
            assert etree.tostring(pa.find('w:pPr',ns))==etree.tostring(pb.find('w:pPr',ns)),('cover paragraph format',i)
            ra,rb=pa.findall('w:r',ns),pb.findall('w:r',ns)
            assert len(ra)==len(rb),('cover run count',i)
            for j,(u,v) in enumerate(zip(ra,rb)):
                up,vp=u.find('w:rPr',ns),v.find('w:rPr',ns)
                assert (etree.tostring(up) if up is not None else None)==(etree.tostring(vp) if vp is not None else None),('cover run format',i,j)
        # Teacher's body rules apply to all prose, including Latin text.
        styles=etree.fromstring(final.read('word/styles.xml'))
        for name,font in [('CourseBody','宋体'),('CourseHeading','黑体'),('CourseSubheading','黑体'),('CourseCaption','宋体'),('CourseReference','宋体')]:
            st=styles.find(f'w:style[@w:styleId="{name}"]',ns)
            fonts=st.find('w:rPr/w:rFonts',ns)
            assert all(fonts.get(qn('w:'+script))==font for script in ['ascii','hAnsi','eastAsia','cs'])
            spacing=st.find('w:pPr/w:spacing',ns)
            assert spacing.get(qn('w:line'))=='240' and spacing.get(qn('w:lineRule'))=='auto'
            assert spacing.get(qn('w:after'))=='0'
            if name in ['CourseHeading','CourseSubheading']:
                assert st.find('w:rPr/w:sz',ns).get(qn('w:val'))=='24'
                assert spacing.get(qn('w:beforeLines'))=='50'
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest()==inventory['sha256']
    return figures

def main():
    global OUTPUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output:OUTPUT=args.output.resolve()
    text,records=content();(HERE/'report-draft.md').write_text(text,encoding='utf-8')
    figures=build(text)
    inputs=[REFERENCE,HERE/'report-content.md',Path(__file__),CORNELL/'parameters.json',CORNELL/'train.yaml',CORNELL/'status.json',CORNELL/'report/learning-analysis.json']
    inputs += [CORNELL/'evaluations'/m['label']/'metrics.json' for m in records]
    inputs += [CORNELL/'evaluations/pretrained/metrics.json',STUDY/'site/docs/assets/course-control/cornell-measurements.json']
    inputs += [STUDY/'experiments/sponza/reference/settings.json',STUDY/'experiments/sponza/reference/pass-validation.json']
    inputs += [STUDY/x['path'] for x in figures]
    inputs += [STUDY/'paper_training/evaluate_cornell.py',STUDY/'paper_training/cornell_learning_analysis.py',
               STUDY/'author_studio/src/renderformer/training/train_rf1.py',
               STUDY/'author_studio/src/renderformer/training/trainer_config.py']
    manifest=dict(snapshot='2026-11-03',kind='stage_report',final_test_evaluated=False,best_step=10000,
        selection='minimum mean validation Reinhard RGB MSE among evaluated archived checkpoints',images_per_checkpoint=128,
        figure_area_cm2=sum(x['area_cm2'] for x in figures),figures=figures,
        input_sha256={str(p.relative_to(STUDY)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        output_sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),preserve_parts_verified=True,
        cover_untouched_slots_verified=True,cover_all_run_and_paragraph_formats_verified=True,
        cover_all_original_paragraphs_identical=True,body_page_number_start=1,cover_page_number_visible=False,
        fresh_template_copy_verified=True,body_teacher_format_verified=True,
        layout_check='pending Word export and page PNG inspection')
    manifest['personalized_cover']=False
    (QA/'build-verification.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Built {OUTPUT}; verified {len(records)} complete validation checkpoints.')

if __name__=='__main__':main()
