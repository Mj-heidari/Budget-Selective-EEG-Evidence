"""Build the shared Windows guide from page-delimited Markdown (requires reportlab)."""
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Preformatted,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.colors import HexColor,white
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

root=Path(__file__).resolve().parents[1]
source=root/'docs/Windows_Run_Guide.md';destination=root/'docs/Windows_Run_Guide.pdf'
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=24,leading=29,textColor=HexColor('#15344c'),spaceAfter=18))
styles.add(ParagraphStyle(name='SubCustom',fontName='Helvetica-Bold',fontSize=13,leading=17,textColor=HexColor('#147f83'),spaceAfter=12))
styles.add(ParagraphStyle(name='TextCustom',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9))
styles.add(ParagraphStyle(name='BulletCustom',parent=styles['TextCustom'],leftIndent=12,firstLineIndent=-8,spaceAfter=6))
styles.add(ParagraphStyle(name='CodeCustom',fontName='Courier',fontSize=8.1,leading=11,backColor=HexColor('#f0f4f7'),borderPadding=8,spaceBefore=4,spaceAfter=12))

def inline(text):
    text=escape(text)
    return re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',text)

def decorate(canvas,doc):
    canvas.setFillColor(HexColor('#15344c'));canvas.rect(0,805,595,37,fill=1,stroke=0)
    canvas.setFillColor(white);canvas.setFont('Helvetica-Bold',9)
    canvas.drawString(44,819,'EEG RESEARCH WORKFLOW  /  REPO3 + REPO4')
    canvas.setFillColor(HexColor('#627381'));canvas.setFont('Helvetica',8)
    canvas.drawString(44,25,'Windows / Python 3.12  |  13 September 2026')
    canvas.drawRightString(551,25,str(doc.page))

story=[]
for page_index,page in enumerate(source.read_text().split('<!-- PAGE -->')):
    if page_index:story.append(PageBreak())
    lines=page.strip().splitlines();i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('```'):
            code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
            maxlen=max(map(len,code),default=1)
            style=ParagraphStyle('CodeFit',parent=styles['CodeCustom'],fontSize=min(8.1,490/max(maxlen,1)/.6))
            story.append(Preformatted('\n'.join(code),style));i+=1;continue
        if line.startswith('# '):story.append(Paragraph(inline(line[2:]),styles['TitleCustom']));i+=1;continue
        if line.startswith('## '):story.append(Paragraph(inline(line[3:]),styles['SubCustom']));i+=1;continue
        if line.startswith('- '):story.append(Paragraph('• '+inline(line[2:]),styles['BulletCustom']));i+=1;continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','- ','```')):
            paragraph.append(lines[i]);i+=1
        story.append(Paragraph(inline(' '.join(paragraph)),styles['TextCustom']))
SimpleDocTemplate(str(destination),pagesize=(595,842),leftMargin=44,rightMargin=44,topMargin=58,bottomMargin=46,
                  title='Windows Run Guide - Repo3 and Repo4',author='Research project documentation').build(story,onFirstPage=decorate,onLaterPages=decorate)
print(destination)
