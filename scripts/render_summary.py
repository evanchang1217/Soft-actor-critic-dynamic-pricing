"""Render the English Markdown research brief as two US Letter PDF pages.

Uses Matplotlib's bundled DejaVu Sans; --font accepts another local font file.
Text and numeric claims come from the Markdown.
"""

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, findfont, fontManager


WIDTH, HEIGHT, MARGIN = 612, 792, 44
BREAK = "<!-- pagebreak -->"


def plain(text):
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.replace("**", "").replace("`", "")


def join_lines(lines):
    return " ".join(lines)


def blocks(text):
    result, paragraph = [], []
    for line in text.splitlines() + [""]:
        hard_break = line.endswith("  ")
        line = line.strip()
        special = line.startswith(("#", "|", "- ")) or re.match(r"^\d+\. ", line)
        if (not line or special) and paragraph:
            result.append(("text", join_lines(paragraph)))
            paragraph = []
        if not line:
            continue
        if line.startswith("|"):
            if re.fullmatch(r"[|\s:\-]+", line):
                continue
            row = [plain(cell.strip()) for cell in line.strip("|").split("|")]
            if result and result[-1][0] == "table":
                result[-1][1].append(row)
            else:
                result.append(("table", [row]))
        elif line.startswith("#"):
            result.append(("title" if line.startswith("# ") else "heading", line.lstrip("# ")))
        else:
            paragraph.append(line)
            if hard_break:
                result.append(("text", join_lines(paragraph)))
                paragraph = []
    return result


def render(source, output, font_path, overview, revision_date="2026-11-06",
           repository_url="https://github.com/evanchang1217/Soft-actor-critic-dynamic-pricing/tree/research/reproducible-sac"):
    date.fromisoformat(revision_date)
    fontManager.addfont(font_path)
    font_family = FontProperties(fname=font_path).get_name()
    def font(size):
        return FontProperties(family=font_family, size=size)
    text = source.read_text(encoding="utf-8")
    if "<!-- brief:start -->" in text:
        text = text.split("<!-- brief:start -->", 1)[1].split("<!-- brief:end -->", 1)[0].strip()
    title = next(line[2:].strip() for line in text.splitlines() if line.startswith("# "))
    first, second = text.split(BREAK, 1)
    pages = (first, second)
    figures = []
    for page_number, content in enumerate(pages, 1):
        fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), dpi=72, facecolor="white")
        renderer = fig.canvas.get_renderer()
        y = HEIGHT - MARGIN
        for kind, value in blocks(content):
            if kind == "table":
                height = 22 * len(value)
                ax = fig.add_axes([MARGIN / WIDTH, (y - height) / HEIGHT,
                                   (WIDTH - 2 * MARGIN) / WIDTH, height / HEIGHT])
                ax.set_axis_off()
                table = ax.table(cellText=value[1:], colLabels=value[0],
                                 colWidths=[0.32, 0.16, 0.18, 0.16, 0.18],
                                 cellLoc="center", bbox=[0, 0, 1, 1])
                table.auto_set_font_size(False)
                for (row, column), cell in table.get_celld().items():
                    cell.set_text_props(fontproperties=font(8))
                    cell.set_edgecolor("#d6dce2")
                    cell.set_linewidth(0.5)
                    if row == 0:
                        cell.set_facecolor("#edf2f6")
                    if column == 0:
                        cell.get_text().set_ha("left")
                y -= height + 10
            else:
                size, leading, gap = {"title": (17, 25, 10), "heading": (12, 19, 7),
                                      "text": (10, 15, 7)}[kind]
                properties = font(size)
                line, width = "", 0
                lines = []
                display = plain(value)
                if (kind == "title" and ": " in display and
                        renderer.get_text_width_height_descent(display, properties, False)[0] > WIDTH - 2 * MARGIN):
                    display = display.replace(": ", ":\n", 1)
                tokens = re.findall(r"[A-Za-z0-9_]+(?:[.,/'-][A-Za-z0-9_]+)*|.", display, re.DOTALL)
                for token in tokens:
                    if token == "\n":
                        lines.append(line.rstrip())
                        line, width = "", 0
                        continue
                    token_width = renderer.get_text_width_height_descent(token, properties, False)[0]
                    closing = token in ",.;:!?)]}%"
                    if width + token_width > WIDTH - 2 * MARGIN and line and not closing:
                        lines.append(line.rstrip())
                        line, width = "", 0
                    line += token
                    width += token_width
                if line:
                    lines.append(line.rstrip())
                for line in lines:
                    fig.text(MARGIN / WIDTH, y / HEIGHT, line, va="top",
                             fontproperties=properties, color="#172b3a")
                    y -= leading
                y -= gap
            if y < MARGIN:
                raise ValueError(f"Brief exceeds page {page_number}; edit the source instead of clipping text")
        if page_number == 1:
            picture = plt.imread(overview)
            image_height = (WIDTH - 2 * MARGIN) * picture.shape[0] / picture.shape[1]
            if y - image_height - 12 < MARGIN:
                raise ValueError("Overview figure does not fit the first page")
            ax = fig.add_axes([MARGIN / WIDTH, (y - image_height - 12) / HEIGHT,
                               (WIDTH - 2 * MARGIN) / WIDTH, image_height / HEIGHT])
            ax.imshow(picture, interpolation="none")
            ax.set_axis_off()
        fig.text(MARGIN / WIDTH, 28 / HEIGHT,
                 f"Research brief | {revision_date} | {page_number} of 2 | Evidence: Markdown brief and README",
                 fontproperties=font(8), color="#536675")
        fig.text(MARGIN / WIDTH, 15 / HEIGHT,
                 "Code and results: github.com/evanchang1217/Soft-actor-critic-dynamic-pricing",
                 fontproperties=font(7.5), color="#4669a3", url=repository_url)
        figures.append(fig)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "project-summary.pdf"
    with PdfPages(path, metadata={"Title": title,
                                  "Author": "evanchang1217 (GitHub)", "CreationDate": None, "ModDate": None}) as pdf:
        for fig in figures:
            pdf.savefig(fig, dpi=180)
            plt.close(fig)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    provenance = {"source": source.as_posix(), "source_sha256": sha(source),
                  "script": "scripts/render_summary.py", "script_sha256": sha(Path(__file__)),
                  "font_file": Path(font_path).name, "font_sha256": sha(Path(font_path)),
                  "latin_font_sha256": sha(Path(findfont("DejaVu Sans"))),
                  "overview": overview.as_posix(), "overview_sha256": sha(overview),
                  "pdf_sha256": sha(path), "pages": 2,
                  "revision_date": revision_date, "repository_url": repository_url,
                  "language": "en", "paper_size": "US Letter"}
    (output / "project-summary.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("docs/project.md"))
    parser.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--font", default=findfont("DejaVu Sans"))
    parser.add_argument("--overview", type=Path, default=Path("docs/assets/control-comparison.png"))
    parser.add_argument("--date", default="2026-11-06", help="ISO revision date printed on the brief")
    parser.add_argument("--repository-url", default="https://github.com/evanchang1217/Soft-actor-critic-dynamic-pricing/tree/research/reproducible-sac")
    args = parser.parse_args()
    if not Path(args.font).is_file():
        parser.error("Provide a local font file with --font")
    render(args.source, args.out, args.font, args.overview, args.date, args.repository_url)
