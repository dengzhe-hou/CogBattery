#!/usr/bin/env python3
"""Check copied data, embedded type, editable SVG and export geometry."""
from pathlib import Path
import csv
import json
import xml.etree.ElementTree as ET

import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "results/reanalysis/human54_layers_20261019"
OUT = ROOT / "paper-overleaf/figures_h54"


def main():
    panel = list(csv.DictReader((DATA / "layer1/panel_differences.csv").open()))
    panel_index = {(r["task"], r["endpoint"]): r for r in panel
                   if r["panel"] == "all54" and r["family_mapping"] == "merged"
                   and r["rule"] == "valid_only" and r["human_cohort"] == "primary"}
    struct = {r["analysis"]: r for r in csv.DictReader((DATA / "layer2/summary.csv").open())}
    report = []
    for stem, expected, height in (("fig_h54_behavior", 25, 99),
                                   ("fig_h54_memory_metacognition", 16, 103),
                                   ("fig_h54_structure", 11, 111)):
        rows = list(csv.DictReader((OUT / f"{stem}_rows.csv").open()))
        assert len(rows) == expected
        for row in rows:
            row.pop("figure_panel")
            source = struct[row["analysis"]] if "analysis" in row else panel_index[(row["task"], row["endpoint"])]
            assert row == source, f"Source values differ: {stem}"
        pdf = fitz.open(OUT / f"{stem}.pdf")
        assert len(pdf) == 1
        page = pdf[0]
        width_mm, height_mm = page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72
        assert abs(width_mm - 180) < .01 and abs(height_mm - height) < .01
        fonts = sorted({f[3] for f in page.get_fonts()})
        assert all("Arial" in f for f in fonts), fonts
        spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block
                 for line in block["lines"] for span in line["spans"]]
        assert spans and min(s["size"] for s in spans) >= 6.99
        assert max(s["size"] for s in spans) <= 8.51
        for span in spans:
            assert page.rect.contains(fitz.Rect(span["bbox"])), (stem, span["text"])
        tree = ET.parse(OUT / f"{stem}.svg")
        text_nodes = tree.findall(".//{http://www.w3.org/2000/svg}text")
        assert text_nodes
        with Image.open(OUT / f"{stem}.png") as im:
            assert all(abs(d - 600) < .1 for d in im.info["dpi"])
            pixels = im.size
        report.append(dict(figure=stem, source_rows_exact=len(rows), width_mm=width_mm,
                           height_mm=height_mm, fonts=fonts,
                           min_font_pt=min(s["size"] for s in spans),
                           max_font_pt=max(s["size"] for s in spans),
                           editable_svg_text_nodes=len(text_nodes), png_pixels=pixels,
                           png_dpi=600, all_pdf_text_within_page=True))
    (OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
