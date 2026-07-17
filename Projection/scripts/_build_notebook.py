#!/usr/bin/env python3
"""Build notebooks/projection_pipeline.ipynb (maintainer utility)."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = PROJECT_ROOT / "notebooks" / "projection_pipeline.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(text),
    }


def _src(text: str) -> list:
    # Keep final newline for nicer diffs; Jupyter accepts list of lines.
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def main() -> None:
    cells: list = []

    cells.append(
        md(
            """# Cryo-EM Projection Pipeline (PDB Biological Assembly → MRC → Projections)

PDB ID から **Biological Assembly** を取得し、EMAN2 で密度マップ化・投影・gallery 作成までを行う研究用ワークフローです。

## 設計方針

| 方針 | 内容 |
|------|------|
| EMAN2 優先 | 密度生成・投影は `e2pdb2mrc.py` / `e2project3d.py` 等を使用。自前アルゴリズムは実装しない |
| 実行方法 | すべての EMAN2 コマンドは `conda run -n eman2 ...` + `subprocess` |
| Assembly | RCSB `*-assembly1.cif` を優先取得 |
| Gemmi | 構造読み込み・粒子径推定・EMAN2 用 PDB 書き出し |
| 再現性 | 実行したコマンド列を最終的に `projection_pipeline.csh` として出力 |

## パイプライン概要

```
PDB ID
  → Biological Assembly (assembly1.cif)
  → Gemmi 読み込み / 粒子径推定 / box size 決定
  → (CIF→PDB) → e2pdb2mrc.py → MRC
  → ヘッダー確認 / COM vs box center / 中央スライス
  → e2project3d.py (Euler 0,0,0 / 90,0,0 / eman:delta=10 gallery)
  → PNG 保存
  → projection_pipeline.csh 出力
```

## 出力ディレクトリ（プロジェクト配下）

```
Projection/
  data/<PDBID>/
    raw/           # assembly CIF
    models/        # Gemmi 書き出し PDB
    maps/          # MRC
    projections/   # HDF
    png/           # PNG
    scripts/       # projection_pipeline.csh
```

**最終目的**: 実験の 2D class average との比較用の参照投影を得る。

### カーネル

- Python 側（gemmi / matplotlib）: プロジェクト `.venv`
- EMAN2: 常に `conda run -n eman2`
"""
        )
    )

    cells.append(
        md(
            """## 0. Setup — imports, paths, EMAN2 runner

Notebook カーネルにはプロジェクトの `.venv`（gemmi / matplotlib）を推奨する。  
EMAN2 本体は常に `conda run -n eman2` 経由で呼ぶ。
"""
        )
    )

    cells.append(
        code(
            r'''# -*- coding: utf-8 -*-
"""
Projection pipeline utilities.
EMAN2 処理は再実装せず、conda run -n eman2 経由で CLI を呼ぶ。
"""
from __future__ import annotations

import math
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project paths (this repository only)
# ---------------------------------------------------------------------------
NOTEBOOK_DIR = Path.cwd().resolve()
if NOTEBOOK_DIR.name == "notebooks":
    PROJECT_ROOT = NOTEBOOK_DIR.parent
else:
    PROJECT_ROOT = Path("/Users/ahgsur/github/structure_tools/Projection")

DATA_ROOT = PROJECT_ROOT / "data"
CONDA_ENV = "eman2"

# FFT-friendly box sizes commonly used in cryo-EM (prefer factors of 2,3,5)
FFT_FRIENDLY_SIZES: Tuple[int, ...] = (
    64, 96, 128, 160, 192, 224, 256, 288, 320, 360, 384,
    432, 448, 480, 512, 576, 640, 720, 768, 864, 960, 1024,
    1152, 1280, 1536, 2048,
)

# Commands executed via EMAN2 (and related) are collected for .csh export
PIPELINE_COMMANDS: List[str] = []


def eman2_cmd(*args: str) -> List[str]:
    """Build argv: conda run -n eman2 -- <eman2-program> ...

    The '--' separator is mandatory to prevent conda from absorbing
    flags like --verbose=1 that belong to the EMAN2 program.
    """
    return ["conda", "run", "-n", CONDA_ENV, "--", *args]


def shlex_quote(s: str) -> str:
    """Minimal POSIX quoting for csh/script export."""
    if not s:
        return "''"
    if all(c.isalnum() or c in "/._-+=:@" for c in s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def run_cmd(
    argv: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    check: bool = True,
    record: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run an external command.

    record=True のとき、後で projection_pipeline.csh に書き出すために
    PIPELINE_COMMANDS へ追記する。
    """
    argv = list(argv)
    print("\n$", " ".join(argv))
    if record:
        PIPELINE_COMMANDS.append(" ".join(shlex_quote(a) for a in argv))

    completed = subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        # conda run は進捗を stderr に出すことが多い
        print(completed.stderr.rstrip(), file=sys.stderr)

    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {completed.returncode}): {' '.join(argv)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def run_eman2(*args: str, **kwargs) -> subprocess.CompletedProcess:
    """Shortcut: conda run -n eman2 ..."""
    return run_cmd(eman2_cmd(*args), **kwargs)


def ensure_gemmi():
    try:
        import gemmi  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "gemmi が見つかりません。プロジェクト .venv をカーネルに選ぶか:\n"
            f"  {PROJECT_ROOT}/.venv/bin/pip install gemmi\n"
            "を実行してください。"
        ) from exc


ensure_gemmi()
import gemmi  # noqa: E402

print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"DATA_ROOT    = {DATA_ROOT}")
print(f"gemmi        = {gemmi.__version__}")

# Quick EMAN2 sanity check
run_eman2("e2version.py", record=True)
'''
        )
    )

    cells.append(
        md(
            """## Step 1 — PDB ID と計算パラメータ

実験データ比較を想定し、`APIX` / `RES` / gallery の `DELTA` を明示的に設定する。
"""
        )
    )

    cells.append(
        code(
            r'''# =============================================================================
# User parameters (edit here)
# =============================================================================
PDB_ID = "1TIM"          # example; replace with your target
APIX = 1.0               # Å/voxel for simulated map (match experiment if possible)
RES = 4.0                # map resolution (Å); must be >= 2*APIX for e2pdb2mrc
INCLUDE_HET = True       # pass --het to e2pdb2mrc
CENTER_IN_BOX = True     # pass --center (atomic COM → box center)
SYMMETRY = "c1"          # for gallery; change if assembly has known point-group
GALLERY_DELTA = 10.0     # degrees for --orientgen=eman:delta=...
BOX_PADDING_FACTOR = 1.5 # box size = particle_diameter * factor, then FFT-round-up

# Optional override: set to int to force box size (None = auto)
BOX_SIZE_OVERRIDE: Optional[int] = None

pdb_id = PDB_ID.strip().upper()
assert len(pdb_id) == 4, f"PDB ID must be 4 characters, got: {pdb_id!r}"
assert RES >= 2.0 * APIX, (
    f"EMAN2 e2pdb2mrc requires RES >= 2*APIX (got RES={RES}, APIX={APIX})"
)

# Per-entry working directories
job_dir = DATA_ROOT / pdb_id
dirs = {
    "raw": job_dir / "raw",
    "models": job_dir / "models",
    "maps": job_dir / "maps",
    "projections": job_dir / "projections",
    "png": job_dir / "png",
    "scripts": job_dir / "scripts",
}
for d in dirs.values():
    d.mkdir(parents=True, exist_ok=True)

# Clear previous command log for this notebook run
PIPELINE_COMMANDS.clear()
# Re-record version for the exported script header
run_eman2("e2version.py", record=True)

print(f"Job directory: {job_dir}")
for k, v in dirs.items():
    print(f"  {k:12s} {v}")
'''
        )
    )

    cells.append(
        md(
            """## Step 2 — RCSB から Biological Assembly を取得

`https://files.rcsb.org/download/{PDBID}-assembly1.cif` を優先する。  
失敗時は assembly2… を試し、それでも無ければエラーにする（非対称単位への silent fallback はしない）。
"""
        )
    )

    cells.append(
        code(
            r'''import urllib.request
import urllib.error

ASSEMBLY_URL_TMPL = "https://files.rcsb.org/download/{pdb_id}-assembly{n}.cif"
MAX_ASSEMBLY_INDEX = 5  # try assembly1..5


def download_biological_assembly(pdb_id: str, out_dir: Path) -> Path:
    """
    Download Biological Assembly mmCIF from RCSB.
    Prefer assembly1.cif; do not silently fall back to asymmetric unit.
    """
    last_error = None
    for n in range(1, MAX_ASSEMBLY_INDEX + 1):
        url = ASSEMBLY_URL_TMPL.format(pdb_id=pdb_id, n=n)
        out_path = out_dir / f"{pdb_id}-assembly{n}.cif"
        print(f"Trying: {url}")
        try:
            # Record curl-equivalent for the csh export
            PIPELINE_COMMANDS.append(
                f"curl -fsSL {shlex_quote(url)} -o {shlex_quote(str(out_path))}"
            )
            with urllib.request.urlopen(url, timeout=120) as resp:
                data = resp.read()
            if not data or len(data) < 100:
                raise IOError(f"Empty or tiny response from {url}")
            out_path.write_bytes(data)
            print(f"Saved Biological Assembly: {out_path} ({len(data):,} bytes)")
            return out_path
        except urllib.error.HTTPError as e:
            last_error = e
            print(f"  HTTP {e.code} for assembly{n}, trying next...")
        except Exception as e:
            last_error = e
            print(f"  Failed assembly{n}: {e}")

    raise FileNotFoundError(
        f"Could not download Biological Assembly for {pdb_id} "
        f"(tried assembly1..{MAX_ASSEMBLY_INDEX}). Last error: {last_error}"
    )


assembly_cif = download_biological_assembly(pdb_id, dirs["raw"])
print(f"Using assembly file: {assembly_cif}")
'''
        )
    )

    cells.append(
        md(
            """## Step 3 — Gemmi で assembly 構造を読み込む

Biological Assembly の原子座標を Gemmi で読み、後段の粒子径推定と PDB 書き出しに使う。
"""
        )
    )

    cells.append(
        code(
            r'''def load_assembly(cif_path: Path) -> gemmi.Structure:
    """Read mmCIF Biological Assembly with Gemmi."""
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    print(f"Name     : {st.name}")
    print(f"Models   : {len(st)}")
    print(f"Spacegroup (may be P1 for assembly): {st.find_spacegroup()}")
    n_chains = sum(len(model) for model in st)
    n_atoms = sum(
        1 for model in st for chain in model for res in chain for atom in res
    )
    print(f"Chains   : {n_chains}")
    print(f"Atoms    : {n_atoms}")
    return st


structure = load_assembly(assembly_cif)
'''
        )
    )

    cells.append(
        md(
            """## Step 4 — 粒子径の推定（重心・最大半径・最大直径）

実験粒子の box 設定に合わせるための幾何学的推定。  
水素は除外し、通常原子の座標から COM と最大半径を計算する。
"""
        )
    )

    cells.append(
        code(
            r'''@dataclass
class ParticleMetrics:
    n_atoms: int
    com_A: Tuple[float, float, float]
    max_radius_A: float
    max_diameter_A: float


def estimate_particle_metrics(
    st: gemmi.Structure, model_index: int = 0
) -> ParticleMetrics:
    """
    Center of mass (equal weight per atom), max radius from COM, diameter = 2*R.
    Excludes hydrogens. Uses first model by default.
    """
    model = st[model_index]
    coords = []
    for chain in model:
        for res in chain:
            for atom in res:
                if atom.is_hydrogen():
                    continue
                pos = atom.pos
                coords.append((pos.x, pos.y, pos.z))

    if not coords:
        raise ValueError("No non-hydrogen atoms found in assembly")

    arr = np.asarray(coords, dtype=np.float64)
    com = arr.mean(axis=0)
    radii = np.linalg.norm(arr - com, axis=1)
    r_max = float(radii.max())
    return ParticleMetrics(
        n_atoms=len(coords),
        com_A=(float(com[0]), float(com[1]), float(com[2])),
        max_radius_A=r_max,
        max_diameter_A=2.0 * r_max,
    )


metrics = estimate_particle_metrics(structure)
print(f"Atoms used     : {metrics.n_atoms}")
print(
    f"COM (Å)        : "
    f"({metrics.com_A[0]:.3f}, {metrics.com_A[1]:.3f}, {metrics.com_A[2]:.3f})"
)
print(f"Max radius (Å) : {metrics.max_radius_A:.2f}")
print(f"Max diameter(Å): {metrics.max_diameter_A:.2f}")
'''
        )
    )

    cells.append(
        md(
            """## Step 5 — Box size 決定

`box_Å = particle_diameter × 1.5` を voxel に直し、FFT-friendly size へ切り上げる。
"""
        )
    )

    cells.append(
        code(
            r'''def next_fft_friendly(n: int) -> int:
    """Round up to the next FFT-friendly box size."""
    n = int(math.ceil(n))
    for s in FFT_FRIENDLY_SIZES:
        if s >= n:
            return s
    # fallback: next multiple of 16
    return int(math.ceil(n / 16.0) * 16)


box_edge_A = metrics.max_diameter_A * BOX_PADDING_FACTOR
box_pixels_raw = box_edge_A / APIX

if BOX_SIZE_OVERRIDE is not None:
    box_size = int(BOX_SIZE_OVERRIDE)
    print(f"Using BOX_SIZE_OVERRIDE = {box_size}")
else:
    box_size = next_fft_friendly(box_pixels_raw)

print(f"Particle diameter     : {metrics.max_diameter_A:.2f} Å")
print(f"Padding factor        : {BOX_PADDING_FACTOR}")
print(
    f"Requested box edge    : {box_edge_A:.2f} Å  "
    f"({box_pixels_raw:.1f} px @ {APIX} Å/px)"
)
print(f"FFT-friendly box size : {box_size}^3")
print(f"Physical box edge     : {box_size * APIX:.2f} Å")
'''
        )
    )

    cells.append(
        md(
            """## Step 6 — EMAN2 で PDB/mmCIF → MRC（`e2pdb2mrc.py`）

`e2pdb2mrc.py` は古典 PDB パーサを前提とするため、Gemmi で assembly を PDB に書き出してから変換する。  
密度計算自体は EMAN2 に委譲する（`--center` で原子 COM を box 中心へ）。
"""
        )
    )

    cells.append(
        code(
            r'''# Write PDB for EMAN2 (Biological Assembly coordinates)
assembly_pdb = dirs["models"] / f"{pdb_id}_assembly1.pdb"
structure.write_pdb(str(assembly_pdb))
print(f"Wrote PDB for EMAN2: {assembly_pdb}")
PIPELINE_COMMANDS.append(
    f"# Gemmi write_pdb -> {assembly_pdb} "
    f"(CLI equivalent: gemmi convert {assembly_cif} {assembly_pdb})"
)

mrc_path = (
    dirs["maps"]
    / f"{pdb_id}_assembly1_apix{APIX}_res{RES}_box{box_size}.mrc"
)

pdb2mrc_args = [
    "e2pdb2mrc.py",
    str(assembly_pdb),
    str(mrc_path),
    f"--apix={APIX}",
    f"--res={RES}",
    f"--box={box_size}",
]
if INCLUDE_HET:
    pdb2mrc_args.append("--het")
if CENTER_IN_BOX:
    pdb2mrc_args.append("--center")

run_eman2(*pdb2mrc_args)
assert mrc_path.is_file(), f"MRC not created: {mrc_path}"
print(f"MRC written: {mrc_path}")
'''
        )
    )

    cells.append(
        md(
            """## Step 7 — MRC ヘッダー確認（voxel size / box size / density range）

`e2iminfo.py` で統計を取得し、必要に応じて EMAN2 Python API でも確認する。
"""
        )
    )

    cells.append(
        code(
            r'''# Header + statistics via EMAN2 CLI
# NOTE: e2iminfo.py allows only one of --header (-H) or --stat (-s) at a time.
print("=== MRC header (-H) ===")
run_eman2("e2iminfo.py", str(mrc_path), "-H")
print("\n=== MRC statistics (-s) ===")
run_eman2("e2iminfo.py", str(mrc_path), "-s")

# Structured summary via EMAN2 Python (still EMAN2, not a custom MRC parser)
inspect_script = dirs["scripts"] / "_inspect_mrc_header.py"
inspect_script.write_text(
    textwrap.dedent(
        """\
        from EMAN2 import EMData
        import json
        e = EMData(r"{mrc}", 0)
        info = {{
            "nx": e["nx"], "ny": e["ny"], "nz": e["nz"],
            "apix_x": e["apix_x"], "apix_y": e["apix_y"], "apix_z": e["apix_z"],
            "minimum": e["minimum"], "maximum": e["maximum"],
            "mean": e["mean"], "sigma": e["sigma"],
        }}
        print(json.dumps(info, indent=2))
        """
    ).format(mrc=str(mrc_path))
)
print("\n=== Structured summary (EMAN2 EMData) ===")
run_eman2("python", str(inspect_script))
'''
        )
    )

    cells.append(
        md(
            """## Step 8 — MRC 重心確認（density COM vs box center）

密度の center of mass は EMAN2 の `EMData.calc_center_of_mass` を使用する。  
`--center` 付きで生成していれば、COM は box 中心付近にあるはず。
"""
        )
    )

    cells.append(
        code(
            r'''com_script = dirs["scripts"] / "_check_com.py"
com_script.write_text(
    textwrap.dedent(
        """\
        from EMAN2 import EMData
        import json

        e = EMData(r"{mrc}", 0)
        nx, ny, nz = e["nx"], e["ny"], e["nz"]
        # EMAN2 requires a density threshold (float) for COM
        thr = max(0.0, float(e["mean"]))
        com = e.calc_center_of_mass(thr)
        box_center_half = [nx / 2.0, ny / 2.0, nz / 2.0]
        delta = [com[i] - box_center_half[i] for i in range(3)]
        delta_A = [delta[i] * e["apix_x"] for i in range(3)]
        out = {{
            "density_com_px": [float(com[0]), float(com[1]), float(com[2])],
            "box_center_half_px": box_center_half,
            "delta_px": [float(d) for d in delta],
            "delta_A": [float(d) for d in delta_A],
            "apix": float(e["apix_x"]),
            "box": [nx, ny, nz],
        }}
        print(json.dumps(out, indent=2))
        r = (delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2) ** 0.5
        if r > 1.0:
            print(f"WARNING: density COM is {{r:.2f}} px from box center")
        else:
            print(f"OK: density COM within {{r:.3f}} px of box center")
        """
    ).format(mrc=str(mrc_path))
)
run_eman2("python", str(com_script))
'''
        )
    )

    cells.append(
        md(
            """## Step 9 — 中央スライス可視化（XY / XZ / YZ）

マップ中心付近の直交スライスを表示し、センタリングと密度の広がりを目視確認する。
"""
        )
    )

    cells.append(
        code(
            r'''slice_npz = dirs["png"] / f"{pdb_id}_slices.npz"
slice_script = dirs["scripts"] / "_export_volume_numpy.py"
slice_script.write_text(
    textwrap.dedent(
        """\
        from EMAN2 import EMData
        import numpy as np

        e = EMData(r"{mrc}", 0)
        vol = e.numpy().copy()  # typically (nz, ny, nx)
        np.savez(r"{out}", vol=vol, apix=e["apix_x"])
        print("saved", vol.shape, float(vol.min()), float(vol.max()))
        """
    ).format(mrc=str(mrc_path), out=str(slice_npz))
)
run_eman2("python", str(slice_script))

npz = np.load(slice_npz)
vol = npz["vol"]
nz, ny, nx = vol.shape
cz, cy, cx = nz // 2, ny // 2, nx // 2

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
slices = [
    ("XY (z mid)", vol[cz, :, :]),
    ("XZ (y mid)", vol[:, cy, :]),
    ("YZ (x mid)", vol[:, :, cx]),
]
for ax, (title, img) in zip(axes, slices):
    im = ax.imshow(img, cmap="gray", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("px")
    ax.set_ylabel("px")
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(f"{pdb_id} central slices — {mrc_path.name}")
fig.tight_layout()

slice_png = dirs["png"] / f"{pdb_id}_central_slices.png"
fig.savefig(slice_png, dpi=150)
plt.show()
print(f"Saved: {slice_png}")
'''
        )
    )

    cells.append(
        md(
            """## Step 10 — Projection 生成（`e2project3d.py`）

まず参照用に単一 Euler:

1. `(alt, az, phi) = (0, 0, 0)`
2. `(alt, az, phi) = (90, 0, 0)`

EMAN2 の `--orientgen=single:alt=...:az=...:phi=...` を使用する。
"""
        )
    )

    cells.append(
        code(
            r'''proj_000 = dirs["projections"] / f"{pdb_id}_proj_alt0_az0_phi0.hdf"
proj_900 = dirs["projections"] / f"{pdb_id}_proj_alt90_az0_phi0.hdf"

for out, alt, az, phi in [
    (proj_000, 0.0, 0.0, 0.0),
    (proj_900, 90.0, 0.0, 0.0),
]:
    if out.exists():
        out.unlink()
    run_eman2(
        "e2project3d.py",
        str(mrc_path),
        f"--outfile={out}",
        f"--orientgen=single:alt={alt}:az={az}:phi={phi}",
        f"--sym={SYMMETRY}",
        "--projector=standard",
    )
    print(f"Wrote projection Euler=({alt},{az},{phi}): {out}")

print("\n--- proj Euler=(0,0,0) ---")
run_eman2("e2iminfo.py", str(proj_000), "-s")
print("\n--- proj Euler=(90,0,0) ---")
run_eman2("e2iminfo.py", str(proj_900), "-s")
'''
        )
    )

    cells.append(
        md(
            """## Step 11 — Projection gallery（`eman:delta=10`）

準均一な向きサンプルを生成し、2D class average 比較用の gallery とする。
"""
        )
    )

    cells.append(
        code(
            r'''gallery_hdf = (
    dirs["projections"]
    / f"{pdb_id}_gallery_eman_delta{int(GALLERY_DELTA)}.hdf"
)
if gallery_hdf.exists():
    gallery_hdf.unlink()

run_eman2(
    "e2project3d.py",
    str(mrc_path),
    f"--outfile={gallery_hdf}",
    f"--orientgen=eman:delta={GALLERY_DELTA}:inc_mirror=0",
    f"--sym={SYMMETRY}",
    "--projector=standard",
)

# Image count (e2iminfo -c: count only)
run_eman2("e2iminfo.py", str(gallery_hdf), "-c")
# Euler angle listing per image (-a -E shows all images with Euler angles)
run_eman2("e2iminfo.py", str(gallery_hdf), "-a", "-E")
'''
        )
    )

    cells.append(
        md(
            """## Step 12 — PNG 保存

EMAN2 `e2proc2d.py` で HDF → PNG へ変換し、gallery は `--unstacking` で連番出力する。  
Notebook 表示用に先頭フレームの montage も作成する。
"""
        )
    )

    cells.append(
        code(
            r'''# --- Single Euler projections → PNG via EMAN2 e2proc2d ---
png_000 = dirs["png"] / f"{pdb_id}_proj_alt0_az0_phi0.png"
png_900 = dirs["png"] / f"{pdb_id}_proj_alt90_az0_phi0.png"

for hdf, png in [(proj_000, png_000), (proj_900, png_900)]:
    run_eman2("e2proc2d.py", str(hdf), str(png), "--outtype=png")

# --- Gallery: write individual PNGs via EMAN2 Python API ---
# Note: e2proc2d --unstacking has a path-numbering bug in EMAN2 2.99.x on macOS.
# We use EMData.write_image() directly instead, which is the EMAN2-native approach.
gallery_png_dir = dirs["png"] / f"{pdb_id}_gallery_delta{int(GALLERY_DELTA)}"
gallery_png_dir.mkdir(exist_ok=True)

preview_npz = dirs["png"] / f"{pdb_id}_gallery_preview.npz"
gallery_write_script = dirs["scripts"] / "_gallery_write_pngs.py"
gallery_write_script.write_text(
    textwrap.dedent(
        """\
        from EMAN2 import EMData
        import numpy as np
        import os

        imgs = EMData.read_images(r"{hdf}")
        outdir = r"{outdir}"
        os.makedirs(outdir, exist_ok=True)
        for i, img in enumerate(imgs):
            img.write_image(os.path.join(outdir, f"proj_{{i:04d}}.png"))
        print(f"Wrote {{len(imgs)}} PNGs to {{outdir}}")

        # Save preview stack for montage
        nshow = min(16, len(imgs))
        stack = np.stack([imgs[i].numpy().copy() for i in range(nshow)], axis=0)
        np.savez(r"{npz}", stack=stack, ntotal=len(imgs))
        print(f"Preview saved: {{nshow}} of {{len(imgs)}}")
        """
    ).format(
        hdf=str(gallery_hdf),
        outdir=str(gallery_png_dir),
        npz=str(preview_npz),
    )
)
run_eman2("python", str(gallery_write_script))

# --- Montage of first N projections for notebook display ---
preview = np.load(preview_npz)
stack = preview["stack"]
ntotal = int(preview["ntotal"])
nshow = stack.shape[0]
ncols = 4
nrows = int(math.ceil(nshow / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.5 * nrows))
axes = np.atleast_2d(axes)
for i in range(nrows * ncols):
    r, c = divmod(i, ncols)
    ax = axes[r, c]
    ax.axis("off")
    if i < nshow:
        ax.imshow(stack[i], cmap="gray", origin="lower")
        ax.set_title(f"#{i}", fontsize=8)
fig.suptitle(
    f"{pdb_id} gallery preview (showing {nshow}/{ntotal}, delta={GALLERY_DELTA}°)"
)
fig.tight_layout()
montage_path = dirs["png"] / f"{pdb_id}_gallery_montage.png"
fig.savefig(montage_path, dpi=150)
plt.show()
print(f"Saved montage: {montage_path}")
print(f"Single PNGs: {png_000}")
print(f"             {png_900}")
print(f"Gallery PNGs ({ntotal} total): {gallery_png_dir}")
'''
        )
    )

    cells.append(
        md(
            """## Step 13 — `projection_pipeline.csh` の出力

この Notebook で記録したコマンド列を、再実行可能な C-shell スクリプトとして書き出す。
"""
        )
    )

    cells.append(
        code(
            r'''csh_path = dirs["scripts"] / "projection_pipeline.csh"

header = textwrap.dedent(
    f"""\
    #!/bin/csh -f
    # =============================================================================
    # projection_pipeline.csh
    # Auto-generated by notebooks/projection_pipeline.ipynb
    # Generated: {datetime.now().isoformat(timespec="seconds")}
    #
    # PDB ID      : {pdb_id}
    # Assembly    : Biological Assembly (RCSB assembly CIF)
    # APIX / RES  : {APIX} / {RES}
    # Box size    : {box_size}
    # Gallery     : eman:delta={GALLERY_DELTA}, sym={SYMMETRY}
    #
    # Prerequisites:
    #   - conda env "{CONDA_ENV}" with EMAN2
    #   - curl (for download)
    #   - gemmi (for CIF→PDB if re-running model prep outside the notebook)
    # =============================================================================
    setenv PROJECT_ROOT {PROJECT_ROOT}
    cd $PROJECT_ROOT

    """
)

body_lines = []
for i, cmd in enumerate(PIPELINE_COMMANDS, 1):
    body_lines.append(f"# step recorded #{i}")
    body_lines.append(cmd)
    body_lines.append("")

footer = textwrap.dedent(
    f"""\
    # Optional: regenerate PDB from CIF with gemmi CLI (if not using the notebook):
    # gemmi convert {assembly_cif} {assembly_pdb}

    echo "Done. MRC: {mrc_path}"
    echo "Gallery: {gallery_hdf}"
    """
)

csh_path.write_text(header + "\n".join(body_lines) + "\n" + footer)
csh_path.chmod(0o755)

project_csh = PROJECT_ROOT / "scripts" / "projection_pipeline.csh"
project_csh.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(csh_path, project_csh)

print(f"Wrote: {csh_path}")
print(f"Copied: {project_csh}")
print(f"Recorded commands: {len(PIPELINE_COMMANDS)}")
print("")
print("----- preview (first 30 lines) -----")
print("\n".join(csh_path.read_text().splitlines()[:30]))
'''
        )
    )

    cells.append(
        md(
            """## Summary — 成果物チェックリスト

実行後、以下が揃っていることを確認する。

- [ ] `data/<PDBID>/raw/*-assembly*.cif`
- [ ] `data/<PDBID>/models/*_assembly1.pdb`
- [ ] `data/<PDBID>/maps/*.mrc`
- [ ] COM が box 中心付近
- [ ] `projections/*_proj_alt0*.hdf` / `*_alt90*.hdf`
- [ ] `projections/*_gallery_eman_delta10.hdf`
- [ ] `png/` 以下の PNG
- [ ] `scripts/projection_pipeline.csh`

### 実験 2D class average との比較メモ

- `APIX` / `box_size` を実験粒子に合わせる
- 必要なら gallery の `delta` を細かくする、または `sym=` を点群に合わせる
- CTF・SNR は未適用のクリーンな投影である点に注意（比較時は low-pass 等を別途）
"""
        )
    )

    cells.append(
        code(
            r'''# Final inventory
print(f"Job: {job_dir}")
for p in sorted(job_dir.rglob("*")):
    if p.is_file() and p.suffix.lower() in {
        ".cif", ".pdb", ".mrc", ".hdf", ".png", ".csh", ".npz"
    }:
        print(f"  {p.relative_to(job_dir)}  ({p.stat().st_size:,} bytes)")
'''
        )
    )

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (Projection .venv)",
                "language": "python",
                "name": "projection-venv",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "cells": cells,
    }

    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"Wrote {NB_PATH} with {len(cells)} cells")


if __name__ == "__main__":
    main()
