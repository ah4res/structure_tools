#!/usr/bin/env python3
"""Batch PISA A-E interface analysis using pisa_analysis.ipynb logic."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NOTEBOOK_PATH = Path(__file__).resolve().parent / "pisa_analysis.ipynb"
CHAIN1, CHAIN2 = "A", "E"

PDB_IDS = [
    "6xez",
    "7cxm",
    "7cxn",
    "7rdx",
    "7rdy",
    "7rdz",
    "7re0",
    "7re1",
    "7re2",
    "7re3",
]


def load_notebook_code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK_PATH.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def init_namespace(cells: list[str]) -> dict:
    ns: dict = {"__name__": "__main__"}
    exec(cells[0], ns)  # deps
    exec(cells[1], ns)  # imports + paths
    exec(cells[3], ns)  # functions
    return ns

def process_pdb(pdb_id: str, ns: dict) -> dict | None:
    pdb_id = pdb_id.strip().upper()
    print("\n" + "=" * 60)
    print(f"Processing {pdb_id} — interface {CHAIN1}-{CHAIN2}")
    print("=" * 60)

    ns["PDB_ID"] = pdb_id
    structure_file = ns["download_mmcif"](pdb_id, ns["DATA_DIR"])
    meta = ns["get_structure_metadata"](pdb_id, structure_file)
    chain_df = ns["get_chain_information"](structure_file, pdb_id)

    pisa_info = ns["run_pisa"](structure_file, pdb_id)
    summary = ns["get_interface_summary"](pisa_info["session_name"], pisa_info["cfg_path"])

    hit = summary[
        ((summary["Chain1"] == CHAIN1) & (summary["Chain2"] == CHAIN2))
        | ((summary["Chain1"] == CHAIN2) & (summary["Chain2"] == CHAIN1))
    ]
    if hit.empty:
        print(f"  WARNING: interface {CHAIN1}-{CHAIN2} not found for {pdb_id}")
        print(summary[["ID", "Chain1", "Chain2", "Area", "deltaG"]].to_string(index=False))
        ns["_erase_pisa_session"](pisa_info)
        return None

    print(hit[["ID", "Chain1", "Chain2", "Area", "deltaG"]].to_string(index=False))

    residues = ns["extract_interface_residues"](
        pisa_info["session_name"],
        pisa_info["cfg_path"],
        CHAIN1,
        CHAIN2,
        summary,
    )
    csv_path = ns["export_interface_csv"](
        residues, meta["pdb_id"], CHAIN1, CHAIN2, ns["RESULTS_DIR"]
    )
    fig, png_path = ns["plot_bsa_barplot"](
        residues,
        meta["pdb_id"],
        CHAIN1,
        CHAIN2,
        ns["RESULTS_DIR"],
        chain_df=chain_df,
        dpi=300,
    )
    plt.close(fig)

    ns["_erase_pisa_session"](pisa_info)

    return {"pdb_id": pdb_id, "csv": csv_path, "png": png_path, "rows": len(residues)}


def _erase_pisa_session_impl(ns: dict, pisa_info: dict) -> None:
    subprocess = ns["subprocess"]
    cmd = [str(ns["PISA_EXE"]), pisa_info["session_name"], "-erase", str(pisa_info["cfg_path"])]
    subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    os.chdir(NOTEBOOK_PATH.parent)
    cells = load_notebook_code_cells()
    ns = init_namespace(cells)
    ns["_erase_pisa_session"] = lambda info: _erase_pisa_session_impl(ns, info)

    results = []
    failures = []

    for pdb_id in PDB_IDS:
        try:
            result = process_pdb(pdb_id, ns)
            if result:
                results.append(result)
        except Exception as exc:
            failures.append((pdb_id, exc))
            print(f"FAILED {pdb_id.upper()}: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Completed: {len(results)} / {len(PDB_IDS)}")
    for r in results:
        print(f"  {r['pdb_id']}: {r['csv'].name}, {r['png'].name} ({r['rows']} rows)")
    if failures:
        print("Failures:")
        for pdb_id, exc in failures:
            print(f"  {pdb_id.upper()}: {exc}")
    print(f"Results: {ns['RESULTS_DIR'].resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
