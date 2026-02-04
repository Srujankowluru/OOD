#!/usr/bin/env python3
"""
Export OOD results to 6 markdown tables + 6 table plots (PNG) in /visinf/projects_students/groupL/results.
Each table: Method | AUROC | FPR@95%
Runs: DeepLab/SegFormer × (sanity Road Anomaly, final Mapillary, final WildDash).
"""

import json
import os


# Base paths (script lives in rkowlu/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPLAB_ROOT = os.path.join(SCRIPT_DIR, "DeepLabV3Plus-Pytorch")
SEGFORMER_ROOT = os.path.join(SCRIPT_DIR, "segformer")
RESULTS_DIR = "/visinf/projects_students/groupL/results"

# 6 runs: (output_filename_without_ext, title_line, path_to_results_summary.json)
RUNS = [
    (
        "deeplab_sanity_road_anomaly",
        "DeepLabV3+ — Sanity check (Road Anomaly)",
        os.path.join(DEEPLAB_ROOT, "new_deep_sanity_results", "results_summary.json"),
    ),
    (
        "segformer_sanity_road_anomaly",
        "SegFormer — Sanity check (Road Anomaly)",
        os.path.join(SEGFORMER_ROOT, "results_road_anomaly_sanity_segformer", "results_summary.json"),
    ),
    (
        "deeplab_final_mapillary",
        "DeepLabV3+ — Final run (Mapillary)",
        os.path.join(DEEPLAB_ROOT, "final_run_deeplab", "results_summary.json"),
    ),
    (
        "deeplab_final_wilddash",
        "DeepLabV3+ — Final run (WildDash)",
        os.path.join(DEEPLAB_ROOT, "final_run_deeplab_wilddash", "results_summary.json"),
    ),
    (
        "segformer_final_mapillary",
        "SegFormer — Final run (Mapillary)",
        os.path.join(SEGFORMER_ROOT, "final_run_segformer", "results_summary.json"),
    ),
    (
        "segformer_final_wilddash",
        "SegFormer — Final run (WildDash)",
        os.path.join(SEGFORMER_ROOT, "final_run_segformer_wilddash", "results_summary.json"),
    ),
]


def load_and_sort(json_path):
    """Load results_summary.json and return rows (method, auroc, fpr95) sorted by AUROC descending."""
    with open(json_path, "r") as f:
        data = json.load(f)
    rows = []
    for method, rec in data.items():
        if isinstance(rec, dict) and "auroc" in rec and "fpr_at_95_tpr" in rec:
            rows.append((method, rec["auroc"], rec["fpr_at_95_tpr"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def format_table(rows):
    """Format rows as markdown table: Method | AUROC | FPR@95%"""
    lines = [
        "| Method | AUROC | FPR@95% |",
        "|--------|-------|---------|",
    ]
    for method, auroc, fpr95 in rows:
        lines.append(f"| {method} | {auroc:.4f} | {fpr95:.4f} |")
    return "\n".join(lines)


def plot_table(rows, title, out_path):
    """Draw table as image (no graphs), save PNG. Uses PIL so no matplotlib needed."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    col_labels = ["Method", "AUROC", "FPR@95%"]
    cell_text = [[m, f"{a:.4f}", f"{f:.4f}"] for m, a, f in rows]
    nrows = len(cell_text) + 1
    ncols = 3
    cell_w = 180
    cell_h = 28
    pad = 12
    title_h = 40
    img_w = ncols * cell_w + 2 * pad
    img_h = title_h + nrows * cell_h + 2 * pad
    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        font_title = font
    draw.text((pad, pad), title, fill="black", font=font_title)
    y0 = title_h + pad
    for r in range(nrows):
        y1 = y0 + cell_h
        for c in range(ncols):
            x0 = pad + c * cell_w
            x1 = x0 + cell_w
            draw.rectangle([x0, y0, x1, y1], outline="black", width=1)
            if r == 0:
                text = col_labels[c]
                fill = (60, 60, 60)
            else:
                text = cell_text[r - 1][c]
                fill = "black"
            tx = x0 + cell_w // 2
            ty = y0 + cell_h // 2
            draw.text((tx, ty), text, fill=fill, font=font, anchor="mm")
        y0 = y1
    out_path = os.path.abspath(out_path)
    img.save(out_path)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for filename_stem, title, json_path in RUNS:
        if not os.path.isfile(json_path):
            print(f"Skip (missing): {json_path}")
            continue
        rows = load_and_sort(json_path)
        table = format_table(rows)
        out_path = os.path.join(RESULTS_DIR, f"{filename_stem}.md")
        with open(out_path, "w") as f:
            f.write(f"# {title}\n\n")
            f.write(table)
            f.write("\n")
        print(f"Wrote {out_path}")
        plot_path = os.path.join(RESULTS_DIR, f"{filename_stem}_table.png")
        plot_table(rows, title, plot_path)
        print(f"Wrote {plot_path}")
    print(f"Done. All tables (and table plots) in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
