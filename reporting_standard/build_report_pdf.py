#!/usr/bin/env python3
"""Вёрстка клиентского отчёта из final_client.md в PDF по разделу 9 методики.

Запуск:
    python build_report_pdf.py <final_client.md> <final_client.pdf> [--author "Исполнитель"]

Понимает подмножество Markdown: заголовки #/##/###, абзацы, маркированные и
нумерованные списки, таблицы, цитаты >, ограждённые блоки кода, горизонтальные
линии, **полужирный**, *курсив*, `моноширинный`, экранированную звёздочку \\*.
Первая горизонтальная линия — конец титульной страницы.
"""
import argparse
import os
import re
import sys

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

PAGE_W, PAGE_H = 595.28, 841.89
M_LEFT = M_RIGHT = 48
M_TOP, M_BOTTOM = 62, 48
HEADER_BASELINE = 40
TEXT_W = PAGE_W - M_LEFT - M_RIGHT

C_TITLE = HexColor("#1F3A5F")
C_SUB = HexColor("#2C5985")
C_TEXT = HexColor("#1A1A1A")
C_GREY = HexColor("#666666")
C_GRID = HexColor("#BBBBBB")
C_HEAD = HexColor("#EFEFEF")
C_CODE = HexColor("#F5F5F5")
C_QUOTE = HexColor("#F2F4F6")

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu",
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")]
FONTS = {"DV": "DejaVuSans.ttf", "DV-B": "DejaVuSans-Bold.ttf",
         "DV-I": "DejaVuSans-Oblique.ttf", "DV-BI": "DejaVuSans-BoldOblique.ttf",
         "DV-M": "DejaVuSansMono.ttf"}


def register_fonts():
    for name, fname in FONTS.items():
        for d in FONT_DIRS:
            path = os.path.join(d, fname)
            if os.path.exists(path):
                pdfmetrics.registerFont(TTFont(name, path))
                break
        else:
            sys.exit(f"Шрифт {fname} не найден (искали в {FONT_DIRS})")
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")


def style(name, size, leading, color=C_TEXT, font="DV", align=TA_LEFT, **kw):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=leading,
                          textColor=color, alignment=align, **kw)


S = {}


def build_styles():
    S["title"] = style("title", 22, 27, C_TITLE, "DV-B", spaceAfter=10)
    S["subtitle"] = style("subtitle", 12, 16, C_SUB, "DV-B", spaceAfter=14)
    S["meta"] = style("meta", 9.5, 14, spaceAfter=2)
    S["intro"] = style("intro", 9, 13.3, align=TA_LEFT, spaceBefore=10, spaceAfter=6)
    S["h1"] = style("h1", 15.2, 19, C_TITLE, "DV-B", spaceBefore=14, spaceAfter=8)
    S["h2"] = style("h2", 11.5, 15, C_SUB, "DV-B", spaceBefore=10, spaceAfter=5)
    S["h3"] = style("h3", 10.2, 14, C_SUB, "DV-B", spaceBefore=8, spaceAfter=4)
    S["body"] = style("body", 8.6, 13.3, align=TA_LEFT, spaceAfter=5)
    S["li"] = style("li", 8.6, 13.3, align=TA_LEFT, leftIndent=14, bulletIndent=4, spaceAfter=2)
    S["cell"] = style("cell", 6.1, 8.4)
    S["cellh"] = style("cellh", 6.1, 8.4, font="DV-B")
    S["code"] = style("code", 7.4, 10.4, font="DV-M", backColor=C_CODE, borderPadding=5,
                      spaceBefore=4, spaceAfter=8)
    S["quote"] = style("quote", 9.4, 14, font="DV-B", backColor=C_QUOTE, borderPadding=8,
                       spaceBefore=8, spaceAfter=10, leftIndent=4, rightIndent=4)
    S["closing"] = style("closing", 7.6, 11.5, C_GREY, "DV-I", spaceBefore=12)


ESC_STAR = "\u0000STAR\u0000"


def inline(text):
    """Markdown inline -> разметка Paragraph."""
    text = text.replace("\\*", ESC_STAR)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    codes = []

    def keep_code(m):
        codes.append(m.group(1))
        return f"\u0001{len(codes) - 1}\u0001"

    text = re.sub(r"`([^`]+)`", keep_code, text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"\u0001(\d+)\u0001", lambda m: f'<font name="DV-M">{codes[int(m.group(1))]}</font>', text)
    return text.replace(ESC_STAR, "*")


def plain(text):
    """Текст ячейки без разметки — для расчёта ширины колонок."""
    return re.sub(r"[*`\\]", "", text)


def split_row(line):
    line = line.strip().strip("|")
    cells, cur, esc = [], "", False
    for ch in line:
        if ch == "\\" and not esc:
            esc = True
            cur += ch
            continue
        if ch == "|" and not esc:
            cells.append(cur.strip())
            cur = ""
        else:
            cur += ch
        esc = False
    cells.append(cur.strip())
    return [c.replace("\\|", "|") for c in cells]


def col_widths(rows, total):
    """Минимум — самое длинное слово (с потолком), остаток — пропорционально объёму текста."""
    n = max(len(r) for r in rows)
    pad = 6.0
    cap = total * 0.28
    mins, vols = [], []
    for j in range(n):
        longest, vol = 0.0, 0
        for i, r in enumerate(rows):
            if j >= len(r):
                continue
            font = "DV-B" if i == 0 else "DV"
            txt = plain(r[j])
            for w in txt.split():
                longest = max(longest, stringWidth(w, font, 6.1))
            vol += len(txt)
        mins.append(min(longest + pad, cap))
        vols.append(max(vol, 1))
    base = sum(mins)
    if base >= total:
        return [m * total / base for m in mins]
    rest = total - base
    tv = sum(vols)
    return [m + rest * v / tv for m, v in zip(mins, vols)]


def make_table(rows):
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    widths = col_widths(rows, TEXT_W)
    data = [[Paragraph(inline(c), S["cellh"] if i == 0 else S["cell"]) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, C_GRID),
        ("BACKGROUND", (0, 0), (-1, 0), C_HEAD),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [t, Spacer(1, 8)]


def parse(md):
    lines = md.splitlines()
    flow, title_page = [], True
    meta = {"title": "", "subtitle": ""}
    para = []
    i = 0

    def flush():
        nonlocal para
        if not para:
            return
        text = " ".join(s.strip() for s in para)
        para = []
        if re.fullmatch(r"\*[^*].*[^*]\*", text):
            flow.append(Paragraph(inline(text[1:-1]), S["closing"]))
        elif title_page:
            st = S["meta"] if re.match(r"\*\*[^*]+:\*\*", text) else S["intro"]
            flow.append(Paragraph(inline(text), st))
        else:
            flow.append(Paragraph(inline(text), S["body"]))

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("```"):
            flush()
            buf = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            flow.append(Preformatted("\n".join(buf), S["code"]))
            i += 1
            continue
        if not s:
            flush()
            i += 1
            continue
        if re.fullmatch(r"-{3,}|\*{3,}", s):
            flush()
            if title_page:
                title_page = False
                flow.append(PageBreak())
            else:
                flow.append(Spacer(1, 6))
            i += 1
            continue
        m = re.match(r"(#{1,3})\s+(.*)", s)
        if m:
            flush()
            lvl, txt = len(m.group(1)), m.group(2)
            if title_page:
                if lvl == 1:
                    meta["title"] = plain(txt)
                    flow.append(Paragraph(inline(txt), S["title"]))
                else:
                    meta["subtitle"] = plain(txt)
                    flow.append(Paragraph(inline(txt), S["subtitle"]))
            else:
                flow.append(Paragraph(inline(txt), S[f"h{lvl}"]))
            i += 1
            continue
        if s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                r = split_row(lines[i])
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in r if c):
                    rows.append(r)
                i += 1
            flow.extend(make_table(rows))
            continue
        if s.startswith(">"):
            flush()
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            flow.append(Paragraph(inline(" ".join(buf)), S["quote"]))
            continue
        m = re.match(r"([-*]|\d+[.)])\s+(.*)", s)
        if m:
            flush()
            marker = "•" if m.group(1) in "-*" else m.group(1)
            text = m.group(2)
            i += 1
            while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() \
                    and not re.match(r"\s*([-*]|\d+[.)])\s+", lines[i]):
                text += " " + lines[i].strip()
                i += 1
            flow.append(Paragraph(inline(text), S["li"], bulletText=marker))
            continue
        para.append(line)
        i += 1
    flush()
    return flow, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--author", default="")
    a = ap.parse_args()
    register_fonts()
    build_styles()
    with open(a.src, encoding="utf-8") as f:
        flow, meta = parse(f.read())

    def header(canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        y = PAGE_H - HEADER_BASELINE
        canvas.setFont("DV", 7)
        canvas.setFillColor(C_SUB)
        canvas.drawString(M_LEFT, y, meta["title"])
        canvas.setFillColor(C_GREY)
        canvas.drawRightString(PAGE_W - M_RIGHT, y, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(a.dst, pagesize=(PAGE_W, PAGE_H), leftMargin=M_LEFT,
                            rightMargin=M_RIGHT, topMargin=M_TOP, bottomMargin=M_BOTTOM,
                            title=meta["title"], subject=meta["subtitle"], author=a.author,
                            creator=a.author, producer=a.author)
    doc.build(flow, onFirstPage=header, onLaterPages=header)
    print(f"OK: {a.dst}")


if __name__ == "__main__":
    main()
