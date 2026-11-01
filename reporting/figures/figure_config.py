from __future__ import annotations
from pathlib import Path
import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(os.environ.get("SCAP_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUTPUT_DIR = PROJECT_ROOT / "publication_figures_v4_2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 600

def read_csv_any(basename: str) -> pd.DataFrame:
    hits = sorted(PROJECT_ROOT.rglob(basename), key=lambda p: (len(p.parts), str(p).lower()))
    if hits:
        print(f"[READ] {basename} <- {hits[0]}")
        return pd.read_csv(hits[0], encoding="utf-8-sig")

    for zp in sorted(PROJECT_ROOT.rglob("*.zip")):
        try:
            with zipfile.ZipFile(zp, "r") as z:
                members = [m for m in z.namelist() if Path(m).name == basename]
                if members:
                    member = sorted(members, key=len)[0]
                    print(f"[READ] {basename} <- {zp} :: {member}")
                    with z.open(member) as f:
                        return pd.read_csv(f, encoding="utf-8-sig")
        except zipfile.BadZipFile:
            continue

    raise FileNotFoundError(f"Could not find {basename} below {PROJECT_ROOT}")

def set_pub_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.3,
        "axes.titlesize": 8.9,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save_figure(fig, stem):
    for ext in ("png", "tiff", "pdf"):
        p = OUTPUT_DIR / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext in ("png", "tiff"):
            kwargs["dpi"] = DPI
        fig.savefig(p, **kwargs)
        print(f"[SAVED] {p}")
