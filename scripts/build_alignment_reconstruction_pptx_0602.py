from __future__ import annotations

import html
import textwrap
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PPTX = ROOT / "docs/00_current/EEG-Voice:Audio-对齐设计.pptx"
OUTPUT_PPTX = ROOT / "docs/00_current/EEG-Voice:Audio-对齐与语音重建设计图_0602.pptx"
OUTPUT_SVG = ROOT / "docs/06_assets/eeg_audio_alignment_reconstruction_0602.svg"

SVG_MEDIA_NAME = "eeg_audio_alignment_reconstruction_0602.svg"
SVG_MEDIA_PATH_IN_ZIP = f"ppt/media/{SVG_MEDIA_NAME}"


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def rect(x, y, w, h, rx=18, fill="#ffffff", stroke="#4b4b4b", stroke_width=2):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def text_block(
    x: int,
    y: int,
    w: int,
    lines: list[str],
    font_size: int = 24,
    weight: str = "500",
    fill: str = "#1f1f1f",
    line_gap: int = 30,
):
    start_y = y + (font_size if len(lines) == 1 else font_size - 4)
    parts = [
        (
            f'<text x="{x + w / 2:.1f}" y="{start_y + i * line_gap}" '
            f'font-family="PingFang SC, Microsoft YaHei, Arial, sans-serif" '
            f'font-size="{font_size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="middle">{esc(line)}</text>'
        )
        for i, line in enumerate(lines)
    ]
    return "\n".join(parts)


def lane_label(x: int, y: int, w: int, label: str):
    return (
        rect(x, y, w, 32, rx=12, fill="#f3f4f6", stroke="#c8ccd1", stroke_width=1.5)
        + text_block(x, y + 2, w, [label], font_size=16, weight="600", fill="#4a5565", line_gap=18)
    )


def box(x: int, y: int, w: int, h: int, label: str, font_size: int = 24):
    lines = textwrap.wrap(label, width=18) if len(label) > 18 else [label]
    line_gap = 26 if len(lines) > 1 else 30
    top = y + (h - (len(lines) * line_gap - (line_gap - font_size))) / 2 - 6
    return rect(x, y, w, h) + text_block(x, int(top), w, lines, font_size=font_size, weight="500", line_gap=line_gap)


def arrow(x1: int, y1: int, x2: int, y2: int):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="#6b7280" stroke-width="2.2" marker-end="url(#arrow)"/>'
    )


def generate_svg() -> str:
    width = 1600
    height = 900

    left_x = 90
    center_x = 610
    right_x = 1130
    main_w = 330
    token_w = 330
    center_w = 380

    title = (
        '<text x="800" y="58" font-family="PingFang SC, Microsoft YaHei, Arial, sans-serif" '
        'font-size="34" font-weight="700" fill="#1f2937" text-anchor="middle">'
        "EEG 与 Audio Token 对齐及语音重建设计图"
        "</text>"
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#6b7280"/>',
        "</marker>",
        "</defs>",
        '<rect width="1600" height="900" fill="#ffffff"/>',
        title,
        lane_label(left_x + 45, 86, 240, "EEG tokenization"),
        lane_label(center_x + 80, 86, 220, "Alignment"),
        lane_label(right_x + 40, 86, 250, "Audio tokenization"),
        lane_label(center_x + 45, 760, 290, "Speech reconstruction"),
    ]

    left_boxes = [
        ("EEG signal", 130),
        ("Preprocess / epoch", 205),
        ("EEG representation", 280),
        ("EEG token space", 355),
    ]
    for label, y in left_boxes:
        parts.append(box(left_x, y, main_w, 52, label, font_size=24))
    for (_, y1), (_, y2) in zip(left_boxes, left_boxes[1:]):
        parts.append(arrow(left_x + main_w // 2, y1 + 52, left_x + main_w // 2, y2))

    left_tokens = [
        ("Auditory base", 460),
        ("Speech content", 512),
        ("Prosody", 564),
        ("Voice identity", 616),
        ("Residual / noise", 668),
    ]
    for label, y in left_tokens:
        parts.append(box(left_x, y, token_w, 40, label, font_size=20))
        parts.append(arrow(left_x + main_w // 2, 407, left_x + token_w // 2, y))

    right_boxes = [
        ("Audio / voice bank", 130),
        ("Audio tokenizers", 205),
        ("AudioTokenBundle", 280),
        ("Audio token space", 355),
    ]
    for label, y in right_boxes:
        parts.append(box(right_x, y, main_w, 52, label, font_size=24))
    for (_, y1), (_, y2) in zip(right_boxes, right_boxes[1:]):
        parts.append(arrow(right_x + main_w // 2, y1 + 52, right_x + main_w // 2, y2))

    right_tokens = [
        ("Envelope / onset", 460),
        ("Content units", 512),
        ("F0 / energy / rhythm", 564),
        ("Speaker / timbre / style", 616),
        ("Codec tokens", 668),
    ]
    for label, y in right_tokens:
        parts.append(box(right_x, y, token_w, 40, label, font_size=20))
        parts.append(arrow(right_x + main_w // 2, 407, right_x + token_w // 2, y))

    parts.append(box(center_x, 320, center_w, 64, "Alignment", font_size=28))
    parts.append(box(center_x, 412, center_w, 56, "Aligned token interface", font_size=24))

    for _, y in left_tokens[:-1]:
        parts.append(arrow(left_x + token_w, y + 20, center_x, 352))
    for _, y in right_tokens[:-1]:
        parts.append(arrow(right_x, y + 20, center_x + center_w, 352))
    parts.append(arrow(center_x + center_w // 2, 384, center_x + center_w // 2, 412))

    support_boxes = [
        ("Content sequence", center_x, 510),
        ("Prosody control", center_x + 200, 510),
        ("Candidate voice prior", center_x, 575),
        ("Decoder backend", center_x + 200, 575),
    ]
    for label, x, y in support_boxes:
        parts.append(box(x, y, 180, 46, label, font_size=18))
        parts.append(arrow(x + 90, y + 46, center_x + center_w // 2, 654))

    parts.append(box(center_x, 655, center_w, 54, "Reconstruction interface", font_size=24))
    parts.append(arrow(center_x + center_w // 2, 468, center_x + center_w // 2, 655))
    parts.append(box(center_x + 48, 734, center_w - 96, 50, "Decoder / vocoder", font_size=24))
    parts.append(arrow(center_x + center_w // 2, 709, center_x + center_w // 2, 734))
    parts.append(box(center_x + 78, 805, center_w - 156, 44, "Speech waveform", font_size=22))
    parts.append(arrow(center_x + center_w // 2, 784, center_x + center_w // 2, 805))

    parts.append("</svg>")
    return "\n".join(parts)


SLIDE_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      <p:pic>
        <p:nvPicPr>
          <p:cNvPr id="2" name="Alignment Reconstruction Figure"/>
          <p:cNvPicPr>
            <a:picLocks noChangeAspect="1"/>
          </p:cNvPicPr>
          <p:nvPr/>
        </p:nvPicPr>
        <p:blipFill>
          <a:blip r:embed="rId2"/>
          <a:stretch>
            <a:fillRect/>
          </a:stretch>
        </p:blipFill>
        <p:spPr>
          <a:xfrm>
            <a:off x="182880" y="68580"/>
            <a:ext cx="11826240" cy="6652260"/>
          </a:xfrm>
          <a:prstGeom prst="rect">
            <a:avLst/>
          </a:prstGeom>
        </p:spPr>
      </p:pic>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr>
    <a:masterClrMapping/>
  </p:clrMapOvr>
</p:sld>
"""


SLIDE_RELS_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
                Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout"
                Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2"
                Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
                Target="../media/{SVG_MEDIA_NAME}"/>
</Relationships>
"""


def update_content_types(xml_bytes: bytes) -> bytes:
    ET.register_namespace("", "http://schemas.openxmlformats.org/package/2006/content-types")
    root = ET.fromstring(xml_bytes)
    ns = {"ct": "http://schemas.openxmlformats.org/package/2006/content-types"}
    exists = any(node.attrib.get("Extension") == "svg" for node in root.findall("ct:Default", ns))
    if not exists:
        ET.SubElement(
            root,
            "{http://schemas.openxmlformats.org/package/2006/content-types}Default",
            {"Extension": "svg", "ContentType": "image/svg+xml"},
        )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build_pptx(svg_content: str):
    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SVG.write_text(svg_content, encoding="utf-8")

    with zipfile.ZipFile(TEMPLATE_PPTX, "r") as zin, zipfile.ZipFile(OUTPUT_PPTX, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            if info.filename in {
                "[Content_Types].xml",
                "ppt/slides/slide1.xml",
                "ppt/slides/_rels/slide1.xml.rels",
            }:
                continue
            data = zin.read(info.filename)
            zout.writestr(info, data)

        content_types = update_content_types(zin.read("[Content_Types].xml"))
        zout.writestr("[Content_Types].xml", content_types)
        zout.writestr("ppt/slides/slide1.xml", SLIDE_XML)
        zout.writestr("ppt/slides/_rels/slide1.xml.rels", SLIDE_RELS_XML)
        zout.writestr(SVG_MEDIA_PATH_IN_ZIP, svg_content.encode("utf-8"))


def main():
    if not TEMPLATE_PPTX.exists():
        raise FileNotFoundError(f"Missing template PPTX: {TEMPLATE_PPTX}")
    svg = generate_svg()
    build_pptx(svg)
    print(f"SVG: {OUTPUT_SVG}")
    print(f"PPTX: {OUTPUT_PPTX}")


if __name__ == "__main__":
    main()
