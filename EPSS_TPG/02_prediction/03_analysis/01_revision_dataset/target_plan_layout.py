"""Reflow the current target plan; never reuse overflowing source-PDF tables."""
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

WIDTH = A4[0] - 114
INK = colors.HexColor('#263b46')
LINE = colors.HexColor('#94a8b3')
PALE = colors.HexColor('#edf3f6')

# Embedded Unicode fonts keep verbatim Cyrillic excerpts readable in the PDF.
FONT_ROOT = Path('/usr/share/fonts/truetype/liberation2')
for name, filename in (('PlanSerif', 'LiberationSerif-Regular.ttf'),
                       ('PlanSerifBold', 'LiberationSerif-Bold.ttf')):
    pdfmetrics.registerFont(TTFont(name, str(FONT_ROOT / filename)))
pdfmetrics.registerFontFamily('PlanSerif', normal='PlanSerif', bold='PlanSerifBold',
                              italic='PlanSerif', boldItalic='PlanSerifBold')


def styles():
    body = ParagraphStyle('body', fontName='PlanSerif', fontSize=11,
                          leading=14.5, spaceAfter=8, splitLongWords=True)
    return {
        'body': body,
        'title': ParagraphStyle('title', parent=body, fontName='PlanSerifBold',
                                fontSize=23, leading=27, spaceAfter=14),
        'heading': ParagraphStyle('heading', parent=body, fontName='PlanSerifBold',
                                  fontSize=15, leading=18, spaceBefore=14,
                                  spaceAfter=8, keepWithNext=True),
        'subheading': ParagraphStyle('subheading', parent=body, fontName='PlanSerifBold',
                                     fontSize=12, leading=15, spaceBefore=10,
                                     spaceAfter=7, keepWithNext=True),
        'cell': ParagraphStyle('cell', parent=body, fontSize=9.5, leading=12,
                               spaceAfter=0),
        'caption': ParagraphStyle('caption', parent=body, fontSize=9.5, leading=12,
                                  textColor=colors.HexColor('#424e54'), spaceAfter=12),
        'code': ParagraphStyle('code', parent=body, fontName='Courier', fontSize=9,
                               leading=13, spaceAfter=0),
        'bullet': ParagraphStyle('bullet', parent=body, leftIndent=12,
                                 bulletIndent=1, spaceAfter=5),
        'node': ParagraphStyle('node', fontName='Helvetica', fontSize=9.5,
                               leading=12, alignment=1, splitLongWords=True),
    }


class CheckedTable(Table):
    """Record cell bounds and check paragraph heights during the actual layout."""

    def __init__(self, *args, audit=None, **kwargs):
        self.audit = audit
        super().__init__(*args, **kwargs)

    def draw(self):
        super().draw()
        if self.audit is None:
            return
        for row_index, row in enumerate(self._cellvalues):
            for col_index, value in enumerate(row):
                style = self._cellStyles[row_index][col_index]
                w = self._colWidths[col_index]
                h = self._rowHeights[row_index]
                paragraphs = value if isinstance(value, (list, tuple)) else [value]
                used_height = 0
                for paragraph in paragraphs:
                    if isinstance(paragraph, Paragraph):
                        _, ph = paragraph.wrap(w - style.leftPadding - style.rightPadding, 1e6)
                        used_height += ph
                assert used_height <= h - style.topPadding - style.bottomPadding + .1
                x, y = self.canv.absolutePosition(self._colpositions[col_index],
                                                self._rowpositions[row_index + 1])
                self.audit.append({'kind': 'table_cell', 'page': self.canv.getPageNumber(),
                                   'rect': [x, A4[1] - y - h, x + w, A4[1] - y]})


class Architecture(Flowable):
    """Vector diagrams with measured, wrapped labels inside fixed-size nodes."""

    def __init__(self, kind, audit):
        super().__init__()
        self.kind, self.audit = kind, audit
        self.width = WIDTH
        self.height = 90 if kind == 'direct' else 194 if kind == 'trajectory' else 244
        self.keepWithNext = True

    def draw(self):
        c = self.canv
        node_style = styles()['node']

        def node(x, y, w, h, text):
            p = Paragraph(text, node_style)
            _, ph = p.wrap(w - 18, h - 14)
            assert ph <= h - 14, (self.kind, text)
            c.setFillColor(PALE)
            c.setStrokeColor(LINE)
            c.setLineWidth(.7)
            c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
            p.drawOn(c, x + 9, y + (h - ph) / 2)
            ax, ay = c.absolutePosition(x, y)
            self.audit.append({'kind': 'diagram_box', 'page': c.getPageNumber(),
                               'rect': [ax, A4[1] - ay - h, ax + w, A4[1] - ay]})

        def arrow(x1, y1, x2, y2):
            c.setStrokeColor(INK)
            c.setFillColor(INK)
            c.setLineWidth(.9)
            c.line(x1, y1, x2, y2)
            theta = math.atan2(y2 - y1, x2 - x1)
            path = c.beginPath()
            path.moveTo(x2, y2)
            for angle in (theta + 2.65, theta - 2.65):
                path.lineTo(x2 + 6 * math.cos(angle), y2 + 6 * math.sin(angle))
            path.close()
            c.drawPath(path, fill=1, stroke=0)

        if self.kind == 'direct':
            w = (WIDTH - 44) / 3
            for i, text in enumerate(('Observed EPSS<br/>Days 0-30',
                                      'Learned mapping<br/>from past to future',
                                      'One predicted score<br/>Day-60 EPSS')):
                node(i * (w + 22), 16, w, 58, text)
                if i < 2:
                    arrow((i + 1) * w + i * 22 + 2, 45,
                          (i + 1) * (w + 22) - 2, 45)
        elif self.kind == 'trajectory':
            node(0, 128, 195, 55, 'Observed EPSS Days 0-30<br/>and time-valid text, if used')
            node(WIDTH - 195, 128, 195, 55, 'Encoder and fusion<br/>GRU / TPG-GNN')
            arrow(197, 155, WIDTH - 197, 155)
            node(130, 55, WIDTH - 260, 49, 'Regression head<br/>30 outputs at once')
            arrow(WIDTH - 98, 126, WIDTH / 2, 106)
            node(35, 0, WIDTH - 70, 35, 'Day 31, Day 32, ... , Day 59, Day 60')
            arrow(WIDTH / 2, 53, WIDTH / 2, 37)
        else:
            gap = 32
            w = (WIDTH - gap) / 2
            x2 = w + gap
            node(0, 190, w, 48, 'Past EPSS through the anchor<br/>Values, mask, observation age')
            node(x2, 190, w, 48, 'Time-valid CVE text<br/>Available by the anchor')
            node(0, 117, w, 51, 'Temporal encoder (GRU)<br/>History vector: e.g. 128 values')
            node(x2, 117, w, 51, 'TPG + relation-aware GNN<br/>Graph vector: e.g. 256 values')
            arrow(w / 2, 188, w / 2, 170)
            arrow(x2 + w / 2, 188, x2 + w / 2, 170)
            node(95, 59, WIDTH - 190, 35, 'Concatenate vectors: e.g. 384 values')
            arrow(w / 2, 115, 155, 96)
            arrow(x2 + w / 2, 115, WIDTH - 155, 96)
            output = ('Binary head: probability of future VulnCheck listing<br/>'
                      'Selected horizon: 30, 60 or 90 days') if self.kind == 'kev' else (
                      'Regression head: 384 -> 128 -> 32 -> 1<br/>'
                      'Predicted Day-60 EPSS, bounded to [0,1]')
            node(35, 0, WIDTH - 70, 43, output)
            arrow(WIDTH / 2, 57, WIDTH / 2, 45)


def table_widths(header):
    if len(header) == 6:
        sizes = [111, 65, 65, 76, 64, 100]
    elif header[0] == 'ID':
        sizes = [38, 135, 111, 75, 122] if len(header) == 5 else [36, 207, 238]
    elif header[0] == 'CVE':
        sizes = [103, 53, 57, 57, 211]
    elif header[0] == 'Formulation':
        sizes = [87, 66, 100, 228]
    elif header[0] == 'Split':
        sizes = [68, 180, 233]
    elif 'All entries' in header:
        sizes = [64, 58, 106, 253]
    elif len(header) == 5:
        sizes = [64, 114, 101, 101, 101]
    elif len(header) == 4:
        sizes = [64, 139, 139, 139]
    elif len(header) == 3:
        sizes = [241, 120, 120]
    else:
        raise ValueError(f'No column layout for {header}')
    return [w * WIDTH / sum(sizes) for w in sizes]


def build_plan(markdown, figure, output):
    s, story, audit = styles(), [], []
    lines, i = markdown.splitlines(), 0

    def paragraph(text, style='body', **kwargs):
        return Paragraph(escape(text), s[style], **kwargs)

    def box(contents):
        t = CheckedTable([[contents]], colWidths=[WIDTH], audit=audit)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), PALE),
            ('BOX', (0, 0), (-1, -1), .6, LINE),
            ('LEFTPADDING', (0, 0), (-1, -1), 11),
            ('RIGHTPADDING', (0, 0), (-1, -1), 11),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.extend([KeepTogether([t]), Spacer(1, 10)])

    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line == '{{page_break}}':
            story.append(PageBreak())
        elif line.startswith('{{figure_'):
            kind = line[len('{{figure_'):-2]
            if kind == 'epss' or kind.startswith('dated_'):
                files = {
                    'dated_coverage': ('smp_dated_score_coverage.png', 3.7 / 8.5),
                    'dated_trends': ('smp_dated_trend_groups.png', 3.5 / 8.5),
                    'dated_examples': ('smp_dated_real_examples.png', 6.2 / 8.5),
                }
                path, ratio = (figure, 6.1 / 9) if kind == 'epss' else (
                    figure.parent / files[kind][0], files[kind][1])
                image = Image(str(path), width=WIDTH, height=WIDTH * ratio)
                image.keepWithNext = True
                story.append(image)
            else:
                story.append(Architecture(kind, audit))
        elif line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            heading = line[level:].strip()
            if heading.startswith(('10.5 ', '10.6 ')):
                story.append(PageBreak())
            story.append(paragraph(heading, {1: 'title', 2: 'heading', 3: 'subheading'}[level]))
        elif line.startswith('|'):
            rows = []
            i -= 1
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[- :]+', c) for c in cells):
                    rows.append(cells)
                i += 1
            assert all(len(r) == len(rows[0]) for r in rows)
            cells = [[Paragraph('<b>' + escape(c) + '</b>', s['cell']) if ri == 0
                      else paragraph(c, 'cell') for c in row] for ri, row in enumerate(rows)]
            t = CheckedTable(cells, colWidths=table_widths(rows[0]),
                             repeatRows=1, hAlign='LEFT', audit=audit)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), PALE),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), .35, LINE),
                ('LEFTPADDING', (0, 0), (-1, -1), 7),
                ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                ('TOPPADDING', (0, 0), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ]))
            # These tables fit within one page. Keeping each together prevents
            # detached headers and a lone example row at the bottom of a page.
            story.extend([KeepTogether([t]), Spacer(1, 8)])
        elif line.startswith('```'):
            code = []
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(paragraph(lines[i], 'code'))
                i += 1
            i += 1
            box(code)
        else:
            parts = [line]
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|', '- ', '>', '{{', '```')):
                parts.append(lines[i].strip())
                i += 1
            text = ' '.join(parts)
            if text.startswith('> '):
                box(paragraph(text[2:]))
            elif text.startswith('- '):
                story.append(paragraph(text[2:], 'bullet', bulletText='-'))
            else:
                style = 'caption' if text.startswith(('Figure ', 'Table ')) else 'body'
                story.append(paragraph(text, style))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(.4)
        canvas.line(57, 42, A4[0] - 57, 42)
        canvas.setFillColor(INK)
        canvas.setFont('Times-Roman', 8)
        canvas.drawString(57, 29, 'EPSS / TPG-GNN Target Definition and Training Plan')
        canvas.drawRightString(A4[0] - 57, 29, str(doc.page))
        canvas.restoreState()

    # A ReportLab frame has 6-point internal padding; these margins therefore
    # place all content at x=57 with exactly WIDTH points available.
    SimpleDocTemplate(str(output), pagesize=A4, leftMargin=51, rightMargin=51,
                      topMargin=39, bottomMargin=51,
                      title='Target Definition and Training Plan',
                      subject='EPSS forecasting and audited SMP dates; layout corrected',
                      author='').build(story, onFirstPage=footer, onLaterPages=footer)
    return audit


def verify_layout(document, audit):
    """Check every page and every table/diagram box, not only inserted pages."""
    issues = []
    for page_index, page in enumerate(document):
        for word in page.get_text('words'):
            x0, y0, x1, y1, text = word[:5]
            if x0 < 56 or x1 > A4[0] - 56 or y0 < 38 or y1 > A4[1] - 23:
                issues.append({'page': page_index + 1, 'word': text, 'issue': 'page_margin'})
    for box in audit:
        x0, y0, x1, y1 = box['rect']
        for word in document[box['page'] - 1].get_text('words'):
            a, b, c, d, text = word[:5]
            if x0 <= (a+c)/2 <= x1 and y0 <= (b+d)/2 <= y1:
                if a < x0 + 2 or c > x1 - 2 or b < y0 + 1 or d > y1 - 1:
                    issues.append({'page': box['page'], 'word': text, 'issue': box['kind']})
    assert not issues, issues
    return {'all_pages_checked': len(document), 'checked_cells_and_boxes': len(audit),
            'page_margin_issues': 0, 'cell_or_box_overflow_issues': 0}
