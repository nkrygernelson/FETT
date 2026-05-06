"""
Combine per-source CSVs into per-functional raw CSVs for model training.

Run once from project root (after 01_parse_sources.py and 02_query_mp.py):
    uv run python FETT/scripts/data/03_combine_raw.py

Outputs to FETT/data/raw/homemade/ (overwrites existing files):
    pbe.csv    — GGA (PBE):  MP GGA + SNUMAT GGA
    PBESOL.csv — PBEsol:     MP PBEsol + materialscloud PBEsol + javier_home PBEsol
    SCAN.csv   — SCAN:       MP SCAN (includes r2SCAN) + materialscloud SCAN
    GLLBSC.csv — GLLB-SC:    Castelli perovskites
    HSE.csv    — HSE:        MP HSE + SNUMAT HSE
    EXPT.csv   — Experimental: Kingsbury

Note: PBE and PBEsol are now SEPARATE fidelity levels (fidelity IDs 0 and 1).
PBEsol is slightly more accurate than PBE for solids; this split allows the
translation model to learn the PBEsol → higher-fidelity mapping explicitly.

Deduplication across sources:
    1. Assign a sort_fe value to each row:
         FE column (if present) → e_above_hull (if present) → mp_fe_lookup via mpid → NaN
    2. Sort by (sort_fe ascending, source_priority ascending), NaN last.
    3. Drop duplicate formulas keeping the first (= lowest FE, highest-priority source).

Source priorities (lower = preferred when FE is equal or unavailable):
    0 — Materials Project (has FE + e_above_hull)
    1 — materialscloud (no FE; deduped to lowest energy_per_atom in step 01)
    2 — SNUMAT (no FE; deduped to first occurrence in step 01)
    3 — matminer datasets (castelli has FE; kingsbury has mpid for FE lookup)

Output CSV columns: formula, BG  (as expected by make_dataset.py)
"""

import json
import logging
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data"
SOURCE_DIR = DATA_DIR / "interim"
RAW_DIR = DATA_DIR / "raw" / "homemade"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_fe_lookup() -> dict[str, float]:
    path = SOURCE_DIR / "mp_fe_lookup.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    log.warning("mp_fe_lookup.json not found — mpid-based FE lookup disabled")
    return {}


def normalize_formula(formula: str) -> str:
    """Return pymatgen reduced formula, or the original string on failure."""
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return formula


def prepare(df: pd.DataFrame, priority: int, fe_lookup: dict[str, float]) -> pd.DataFrame:
    """
    Normalize formulas, assign a sort_fe column, and tag with source priority.
    sort_fe is NaN if no FE information is available for a row.
    """
    df = df.copy()
    df["formula"] = df["formula"].apply(normalize_formula)

    # Build sort_fe: highest-quality available FE metric per row
    if "FE" in df.columns:
        sort_fe = df["FE"].copy()
    else:
        sort_fe = pd.Series(float("nan"), index=df.index)

    if "e_above_hull" in df.columns:
        mask = sort_fe.isna()
        sort_fe = sort_fe.where(~mask, df.loc[mask, "e_above_hull"])

    if "mpid" in df.columns and fe_lookup:
        mask = sort_fe.isna()
        looked_up = df.loc[mask, "mpid"].map(fe_lookup)
        sort_fe = sort_fe.where(~mask, looked_up)

    df["_sort_fe"] = sort_fe
    df["_priority"] = priority
    return df


def combine_and_save(
    sources: list[tuple[Path, int]],
    fe_lookup: dict[str, float],
    output_path: Path,
    label: str,
) -> None:
    """
    Load source CSVs, prepare, concatenate, deduplicate, and save.
    sources: list of (csv_path, priority) pairs.
    """
    parts = []
    for csv_path, priority in sources:
        if not csv_path.exists():
            log.warning(f"  Source file not found, skipping: {csv_path.name}")
            continue
        df = pd.read_csv(csv_path)
        df = prepare(df, priority, fe_lookup)
        log.info(f"  Loaded {csv_path.name}: {len(df):,} rows (priority {priority})")
        parts.append(df)

    if not parts:
        log.warning(f"No source files found for {label}, skipping {output_path.name}")
        return

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.dropna(subset=["formula", "BG"])
    combined = combined[combined["BG"] >= 0.0]  # band gaps must be non-negative

    before = len(combined)
    combined = (
        combined
        .sort_values(["_sort_fe", "_priority"], na_position="last")
        .drop_duplicates(subset=["formula"])
        .reset_index(drop=True)
    )
    log.info(f"  {before:,} rows → {len(combined):,} unique formulas after deduplication")

    out = combined[["formula", "BG"]].copy()
    out.to_csv(output_path, index=False)
    log.info(f"Saved {output_path.name} ({len(out):,} rows) for {label}\n")


def main():
    fe_lookup = load_fe_lookup()

    # --- PBE / GGA only (standard PBE functional) ---
    # MP GGA+U (Hubbard-corrected GGA) is included at priority 1, after pure GGA.
    # If a formula appears in both GGA and GGA+U, pure GGA is preferred (same dedup rule).
    combine_and_save(
        sources=[
            (SOURCE_DIR / "mp_GGA.csv",    0),  # MP pure GGA: has FE
            (SOURCE_DIR / "mp_GGAU.csv",   1),  # MP GGA+U: has FE (Hubbard-corrected variant)
            (SOURCE_DIR / "snumat_GGA.csv", 2),  # SNUMAT GGA: no FE
        ],
        fe_lookup=fe_lookup,
        output_path=RAW_DIR / "pbe.csv",
        label="PBE/GGA",
    )

    # --- PBEsol (separate fidelity level from PBE) ---
    combine_and_save(
        sources=[
            (SOURCE_DIR / "mp_PBESOL.csv",             0),  # MP PBEsol: has FE
            (SOURCE_DIR / "materialscloud_PBEsol.csv",  1),  # has energy_per_atom, no FE
            (SOURCE_DIR / "javier_pbesol.csv",          2),  # no FE
        ],
        fe_lookup=fe_lookup,
        output_path=RAW_DIR / "PBESOL.csv",
        label="PBEsol",
    )

    # --- SCAN ---
    combine_and_save(
        sources=[
            (SOURCE_DIR / "mp_SCAN.csv",               0),  # MP SCAN: has FE
            (SOURCE_DIR / "materialscloud_SCAN.csv",    1),  # has energy_per_atom, no FE
        ],
        fe_lookup=fe_lookup,
        output_path=RAW_DIR / "SCAN.csv",
        label="SCAN",
    )

    # --- GLLB-SC ---
    combine_and_save(
        sources=[
            (SOURCE_DIR / "castelli_gllbsc.csv",        0),  # has FE (e_form)
        ],
        fe_lookup=fe_lookup,
        output_path=RAW_DIR / "GLLBSC.csv",
        label="GLLB-SC",
    )

    # --- HSE ---
    combine_and_save(
        sources=[
            (SOURCE_DIR / "mp_HSE.csv",                 0),  # MP HSE: has FE
            (SOURCE_DIR / "snumat_HSE.csv",             1),  # no FE; mpid lookup via snumat_id not possible
        ],
        fe_lookup=fe_lookup,
        output_path=RAW_DIR / "HSE.csv",
        label="HSE",
    )

    # --- Experimental ---
    combine_and_save(
        sources=[
            (SOURCE_DIR / "kingsbury_expt.csv",         0),  # has mpid for FE lookup
        ],
        fe_lookup=fe_lookup,
        output_path=RAW_DIR / "EXPT.csv",
        label="Experimental",
    )

    log.info("Done! Raw CSVs written to FETT/data/raw/homemade/")
    log.info("Fidelity levels: pbe(0) pbesol(1) scan(2) gllb-sc(3) hse(4) expt(5)")
    log.info("Run 'uv run invoke make-data' to build processed train/val/test splits.")


if __name__ == "__main__":
    main()
