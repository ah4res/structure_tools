#!/usr/bin/env python3
"""Run rscc_chain_per_residue.ipynb pipeline for selected PDB IDs (chains A/E/F)."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

NOTEBOOK_PATH = Path(__file__).resolve().parent / "rscc_chain_per_residue.ipynb"
OUTPUT_PLOT_DIR = Path(__file__).resolve().parent / "work" / "plots_aef"
TARGET_CHAINS = ["A", "E", "F"]

# 正しい PDB ID（フォルダ名は小文字）
PDB_IDS = [
    "7cxm",
    "7cxn",
]


def load_notebook_code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK_PATH.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def setup_pdb_paths(pdb_id: str) -> dict:
    pdb_id = pdb_id.strip().lower()
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        raise ValueError(f"Invalid PDB ID: {pdb_id!r}")

    work_dir = Path("work") / pdb_id
    work_dir.mkdir(parents=True, exist_ok=True)

    return {
        "PDB_ID": pdb_id,
        "DATA_MODE": "cryoem",
        "LOCAL_MTZ_PATH": None,
        "LOCAL_MAP_PATH": None,
        "RESOLUTION": None,
        "WORK_DIR": work_dir,
        "PDB_PATH": work_dir / f"{pdb_id}.pdb",
        "MTZ_PATH": work_dir / f"{pdb_id}.mtz",
        "SF_CIF_PATH": work_dir / f"{pdb_id}-sf.cif",
        "MAP_PATH": None,
        "CSV_PATH_ATOM": work_dir / f"{pdb_id}_rscc_atom.csv",
        "CSV_PATH_HETATM": work_dir / f"{pdb_id}_rscc_hetatm.csv",
    }


def fix_pdb_download(ns: dict) -> None:
    """RCSB から大文字 PDB ID で PDB を取得（7cxm/7CXM 等の取り違え防止）。"""
    from urllib.request import urlopen

    pdb_id_upper = ns["PDB_ID"].upper()
    url = f"https://files.rcsb.org/download/{pdb_id_upper}.pdb"
    print(f"  Download PDB: {url}")
    with urlopen(url, timeout=120) as resp:
        data = resp.read()
    ns["PDB_PATH"].write_bytes(data)
    print(f"  Saved: {ns['PDB_PATH']} ({len(data):,} bytes)")

    entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id_upper}"
    with urlopen(entry_url, timeout=60) as resp:
        ns["ENTRY_METADATA"] = json.loads(resp.read().decode())
    methods = [x.get("method", "") for x in ns["ENTRY_METADATA"].get("exptl", [])]
    print(f"  Experiment methods: {methods}")


def configure_matplotlib_japanese() -> None:
    from matplotlib import font_manager

    for font_name in ("Hiragino Sans", "Hiragino Kaku Gothic ProN", "Arial Unicode MS", "DejaVu Sans"):
        if any(f.name == font_name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = font_name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
    matplotlib.rcParams["axes.unicode_minus"] = False


def chain_legend_label(chain_id: str, chain_metadata: pd.DataFrame) -> str:
    row = chain_metadata[chain_metadata["chain_id"] == chain_id]
    if row.empty:
        return f"Chain {chain_id}"
    return f"Chain {chain_id} ({row.iloc[0]['chain_description']})"


def plot_rscc_by_residue(dataframe: pd.DataFrame, title: str, ax, chain_metadata: pd.DataFrame) -> None:
    plot_df = dataframe.copy()
    plot_df["residue_num"] = pd.to_numeric(plot_df["residue_number"], errors="coerce")
    for chain_id, group in plot_df.groupby("chain_id", sort=True):
        ax.plot(
            group["residue_num"],
            group["rscc"],
            marker="o",
            markersize=2,
            linewidth=1,
            label=chain_legend_label(str(chain_id), chain_metadata),
        )
    ax.set_xlabel("Residue Number")
    ax.set_ylabel("RSCC")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def run_notebook_pipeline(pdb_id: str, cells: list[str]) -> dict:
    ns = setup_pdb_paths(pdb_id)
    ns["__name__"] = "__main__"

    print(f"PDB ID     : {ns['PDB_ID'].upper()}")
    print(f"Work dir   : {ns['WORK_DIR'].resolve()}")

    exec(cells[2], ns)  # Step 2: tools
    exec(cells[3], ns)  # Step 3: download helpers (skip auto download)
    fix_pdb_download(ns)
    exec(cells[4], ns)  # Step 3.5: chain metadata
    exec(cells[5], ns)  # Step 4: density prep

    atom_csv: Path = ns["CSV_PATH_ATOM"]
    hetatm_csv: Path = ns["CSV_PATH_HETATM"]

    if atom_csv.exists() and atom_csv.stat().st_size > 0:
        print(f"  Skip Phenix (existing CSV): {atom_csv}")
        ns["df_atom"] = pd.read_csv(atom_csv)
        ns["df_hetatm"] = pd.read_csv(hetatm_csv) if hetatm_csv.exists() else pd.DataFrame()
        ns["df"] = pd.concat([ns["df_atom"], ns["df_hetatm"]], ignore_index=True)
    else:
        print("  Running Phenix...")
        exec(cells[6], ns)  # Step 5
        exec(cells[7], ns)  # Step 6
        exec(cells[8], ns)  # Step 7

    return ns


def export_chain_csv(ns: dict) -> Path:
    pdb_id = ns["PDB_ID"]
    selected = ns["df_atom"][ns["df_atom"]["chain_id"].isin(TARGET_CHAINS)].copy()
    out_path = ns["WORK_DIR"] / f"{pdb_id}_rscc_chains_AEF_atom.csv"
    cols = ["chain_id", "chain_description", "chain_residue_count", "residue_number", "residue_name", "rscc"]
    selected[cols].to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(selected)} rows)")
    return out_path


def save_plots(ns: dict, out_dir: Path) -> list[Path]:
    pdb_id = ns["PDB_ID"]
    configure_matplotlib_japanese()
    df_atom = ns["df_atom"]
    meta = ns["CHAIN_METADATA"]
    saved: list[Path] = []

    selected = [c for c in TARGET_CHAINS if c in set(df_atom["chain_id"].astype(str))]
    if selected:
        filtered = df_atom[df_atom["chain_id"].isin(selected)].copy()
        fig, ax = plt.subplots(figsize=(14, 5))
        title = f"ATOM - RSCC per residue ({pdb_id.upper()}) - chains {', '.join(selected)}"
        plot_rscc_by_residue(filtered, title, ax, meta)
        fig.tight_layout()
        combined = out_dir / f"{pdb_id}_rscc_chains_AEF_atom.png"
        fig.savefig(combined, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(combined)
        print(f"  Saved: {combined}")

    for chain_id in TARGET_CHAINS:
        sub = df_atom[df_atom["chain_id"] == chain_id].copy()
        if sub.empty:
            print(f"  WARNING: chain {chain_id} not found")
            continue
        sub["residue_num"] = pd.to_numeric(sub["residue_number"], errors="coerce")
        fig, ax = plt.subplots(figsize=(14, 4.2))
        ax.plot(sub["residue_num"], sub["rscc"], marker="o", markersize=2, linewidth=1, color="#1f77b4")
        ax.set_xlabel("Residue Number")
        ax.set_ylabel("RSCC")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{chain_legend_label(chain_id, meta)} ({pdb_id.upper()})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / f"{pdb_id}_rscc_chain_{chain_id}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        print(f"  Saved: {out_path}")

    return saved


def process_pdb(pdb_id: str, cells: list[str]) -> dict:
    print("\n" + "=" * 60)
    print(f"Processing {pdb_id.upper()}")
    print("=" * 60)
    ns = run_notebook_pipeline(pdb_id, cells)
    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    export_chain_csv(ns)
    save_plots(ns, OUTPUT_PLOT_DIR)
    return ns


def main() -> int:
    os.chdir(NOTEBOOK_PATH.parent)
    cells = load_notebook_code_cells()
    exec(cells[0], {"__name__": "__main__"})

    failures = []
    for pdb_id in PDB_IDS:
        try:
            process_pdb(pdb_id, cells)
        except Exception as exc:
            failures.append((pdb_id, exc))
            print(f"FAILED {pdb_id.upper()}: {exc}")
            traceback.print_exc()

    print(f"\nCompleted: {len(PDB_IDS) - len(failures)} / {len(PDB_IDS)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
