"""
Parse local JSON source files and matminer datasets into per-source CSVs.

Run once from project root:
    uv run python FETT/scripts/data/01_parse_sources.py

Outputs to FETT/data/source/:
    materialscloud_PBEsol.csv  (formula, BG, energy_per_atom)
    materialscloud_SCAN.csv    (formula, BG, energy_per_atom)
    javier_pbesol.csv          (formula, BG)
    snumat_GGA.csv             (snumat_id, formula, BG)
    snumat_HSE.csv             (snumat_id, formula, BG)
    kingsbury_expt.csv         (formula, BG, mpid)
    castelli_gllbsc.csv        (formula, BG, FE)

Deduplication within each source:
    - materialscloud: lowest energy_per_atom per reduced formula (most stable polymorph)
    - SNUMAT:         first occurrence per reduced formula (no FE available)
    - castelli:       lowest e_form per reduced formula
    - kingsbury:      already unique per formula per matminer docs
"""

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data"
JSON_SOURCE_DIR = DATA_DIR / "raw" / "homemade" / "source"
SOURCE_OUT_DIR = DATA_DIR / "interim"
SOURCE_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Run type labels present in the two materialscloud JSON files
MC_PBESOL_RUN_TYPES = {"PBESol"}
MC_SCAN_RUN_TYPES = {"PBE+SCAN", "None or LDA+SCAN"}


def to_reduced_formula(comp) -> str | None:
    """Convert a pymatgen-style composition dict or Counter to a reduced formula string."""
    try:
        return Composition(comp).reduced_formula
    except Exception:
        return None


def parse_materialscloud_json(json_path: Path, valid_run_types: set[str], label: str) -> pd.DataFrame:
    """
    Parse a materials cloud ComputedStructureEntry JSON file.

    Each entry is a pymatgen ComputedStructureEntry with:
        composition: dict of {element: count}
        energy: total DFT energy (eV)
        correction: energy correction (eV)
        parameters.run_type: functional label
        data.eigenvalue_band_properties: [band_gap, cbm, vbm, is_direct]

    Returns a DataFrame with columns: formula, BG, energy_per_atom.
    Deduplicated to one entry per formula (lowest energy_per_atom).
    """
    size_mb = json_path.stat().st_size / 1e6
    log.info(f"Loading {json_path.name} ({size_mb:.0f} MB) ...")
    with open(json_path) as f:
        data = json.load(f)
    entries = data["entries"]
    log.info(f"  {len(entries):,} entries found, parsing {label} ...")

    rows = []
    skipped = 0
    for entry in entries:
        run_type = entry["parameters"].get("run_type", "")
        if run_type not in valid_run_types:
            skipped += 1
            continue

        ebp = entry["data"].get("eigenvalue_band_properties")
        if not ebp:
            skipped += 1
            continue
        bg = ebp[0]

        comp_dict = entry["composition"]
        formula = to_reduced_formula(comp_dict)
        if formula is None:
            skipped += 1
            continue

        n_atoms = sum(comp_dict.values())
        corrected_energy = entry["energy"] + entry.get("correction", 0.0)
        energy_per_atom = corrected_energy / n_atoms

        rows.append({"formula": formula, "BG": bg, "energy_per_atom": energy_per_atom})

    if skipped:
        log.warning(f"  Skipped {skipped:,} entries (wrong run_type or missing data)")

    df = pd.DataFrame(rows)
    log.info(f"  {len(df):,} valid entries → deduplicating to lowest energy_per_atom per formula")
    df = (
        df.sort_values("energy_per_atom")
        .drop_duplicates(subset=["formula"])
        .reset_index(drop=True)
    )
    log.info(f"  {len(df):,} unique formulas for {label}")
    return df


def parse_snumat_json(json_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse the SNUMAT JSON into GGA and HSE DataFrames.

    Each SNUMAT entry has:
        atoms.elements: list of element symbols (one per atom in the unit cell)
        SNUMAT_id: str
        Band_gap_GGA: float (eV)
        Band_gap_HSE: float (eV)

    Returns (df_gga, df_hse), each with columns: snumat_id, formula, BG.
    Deduplicated to first occurrence per reduced formula (no FE available).
    """
    log.info(f"Loading {json_path.name} ...")
    with open(json_path) as f:
        data = json.load(f)
    log.info(f"  {len(data):,} SNUMAT entries found, parsing ...")

    rows = []
    skipped = 0
    for entry in data:
        atoms = entry.get("atoms", {})
        elements = atoms.get("elements") if isinstance(atoms, dict) else None
        if not elements:
            skipped += 1
            continue

        formula = to_reduced_formula(Counter(elements))
        if formula is None:
            skipped += 1
            continue

        rows.append({
            "snumat_id": entry.get("SNUMAT_id", ""),
            "formula": formula,
            "GGA": entry.get("Band_gap_GGA"),
            "HSE": entry.get("Band_gap_HSE"),
        })

    if skipped:
        log.warning(f"  Skipped {skipped:,} SNUMAT entries (missing atoms or unparseable formula)")

    df = pd.DataFrame(rows)
    log.info(f"  {len(df):,} SNUMAT entries parsed")

    df_gga = (
        df[["snumat_id", "formula", "GGA"]]
        .rename(columns={"GGA": "BG"})
        .dropna(subset=["BG"])
        .drop_duplicates(subset=["formula"])
        .reset_index(drop=True)
    )
    df_hse = (
        df[["snumat_id", "formula", "HSE"]]
        .rename(columns={"HSE": "BG"})
        .dropna(subset=["BG"])
        .drop_duplicates(subset=["formula"])
        .reset_index(drop=True)
    )
    log.info(f"  {len(df_gga):,} unique GGA formulas, {len(df_hse):,} unique HSE formulas")
    return df_gga, df_hse


def load_kingsbury_expt() -> pd.DataFrame:
    """
    Load the Kingsbury experimental band gap dataset from matminer.

    Columns in source: formula, expt_gap, likely_mpid
    Output columns: formula, BG, mpid
    """
    log.info("Loading expt_gap_kingsbury from matminer ...")
    from matminer.datasets import load_dataset
    df = load_dataset("expt_gap_kingsbury")
    result = pd.DataFrame({
        "formula": df["formula"],
        "BG": df["expt_gap"],
        "mpid": df["likely_mpid"],
    })
    result = (
        result.dropna(subset=["formula", "BG"])
        .drop_duplicates(subset=["formula"])
        .reset_index(drop=True)
    )
    log.info(f"  {len(result):,} Kingsbury expt entries")
    return result


def load_castelli_gllbsc() -> pd.DataFrame:
    """
    Load the Castelli perovskites GLLB-SC dataset from matminer.

    Columns in source: formula, gap gllbsc, e_form, ...
    Output columns: formula, BG, FE
    FE = e_form (heat of formation per atom, eV)
    Deduplicated to lowest FE per reduced formula.
    """
    log.info("Loading castelli_perovskites from matminer ...")
    from matminer.datasets import load_dataset
    df = load_dataset("castelli_perovskites")
    result = pd.DataFrame({
        "formula": df["formula"].apply(lambda f: to_reduced_formula(f) or f),
        "BG": df["gap gllbsc"],
        "FE": df["e_form"],
    })
    result = (
        result.dropna(subset=["formula", "BG"])
        .sort_values("FE")
        .drop_duplicates(subset=["formula"])
        .reset_index(drop=True)
    )
    log.info(f"  {len(result):,} Castelli GLLB-SC entries")
    return result


def parse_javier_pbesol(csv_path: Path) -> pd.DataFrame:
    """
    Parse the javier_home.csv dataset of PBEsol band gaps.

    Source columns: Material (formula), Pbe_sol (BG eV), HSE (BG eV)
    Output columns: formula, BG  (PBEsol only; HSE values are discarded here)

    Deduplication: keep the entry with the lowest BG per reduced formula
    (proxy for most stable / least excited configuration).
    """
    log.info(f"Loading {csv_path.name} ...")
    df = pd.read_csv(csv_path)

    # Strip whitespace from all string columns
    df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)

    df = df.rename(columns={"Material": "formula", "Pbe_sol": "BG"})
    df = df[["formula", "BG"]].copy()
    df["BG"] = pd.to_numeric(df["BG"], errors="coerce")
    df = df.dropna(subset=["formula", "BG"])
    df["BG"] = df["BG"].clip(lower=0.0)  # band gaps must be non-negative

    # Normalize formulas and deduplicate
    df["formula"] = df["formula"].apply(to_reduced_formula)
    df = df.dropna(subset=["formula"])
    df = (
        df.sort_values("BG")
        .drop_duplicates(subset=["formula"])
        .reset_index(drop=True)
    )
    log.info(f"  {len(df):,} unique PBEsol formulas from {csv_path.name}")
    return df


def main():
    # --- Materials Cloud PBEsol ---
    df_pbesol = parse_materialscloud_json(
        JSON_SOURCE_DIR / "2021.04.06_ps.json",
        valid_run_types=MC_PBESOL_RUN_TYPES,
        label="PBEsol",
    )
    out = SOURCE_OUT_DIR / "materialscloud_PBEsol.csv"
    df_pbesol.to_csv(out, index=False)
    log.info(f"Saved {out.name} ({len(df_pbesol):,} rows)\n")

    # --- Javier PBEsol ---
    javier_path = JSON_SOURCE_DIR / "javier_home.csv"
    if javier_path.exists():
        df_javier = parse_javier_pbesol(javier_path)
        out = SOURCE_OUT_DIR / "javier_pbesol.csv"
        df_javier.to_csv(out, index=False)
        log.info(f"Saved {out.name} ({len(df_javier):,} rows)\n")
    else:
        log.warning(f"javier_home.csv not found at {javier_path} — skipping\n")

    # --- Materials Cloud SCAN ---
    df_scan = parse_materialscloud_json(
        JSON_SOURCE_DIR / "2021.04.06_scan.json",
        valid_run_types=MC_SCAN_RUN_TYPES,
        label="SCAN",
    )
    out = SOURCE_OUT_DIR / "materialscloud_SCAN.csv"
    df_scan.to_csv(out, index=False)
    log.info(f"Saved {out.name} ({len(df_scan):,} rows)\n")

    # --- SNUMAT ---
    df_snumat_gga, df_snumat_hse = parse_snumat_json(JSON_SOURCE_DIR / "snumat.json")
    out = SOURCE_OUT_DIR / "snumat_GGA.csv"
    df_snumat_gga.to_csv(out, index=False)
    log.info(f"Saved {out.name} ({len(df_snumat_gga):,} rows)")
    out = SOURCE_OUT_DIR / "snumat_HSE.csv"
    df_snumat_hse.to_csv(out, index=False)
    log.info(f"Saved {out.name} ({len(df_snumat_hse):,} rows)\n")

    # --- Kingsbury experimental ---
    df_kingsbury = load_kingsbury_expt()
    out = SOURCE_OUT_DIR / "kingsbury_expt.csv"
    df_kingsbury.to_csv(out, index=False)
    log.info(f"Saved {out.name} ({len(df_kingsbury):,} rows)\n")

    # --- Castelli GLLB-SC ---
    df_castelli = load_castelli_gllbsc()
    out = SOURCE_OUT_DIR / "castelli_gllbsc.csv"
    df_castelli.to_csv(out, index=False)
    log.info(f"Saved {out.name} ({len(df_castelli):,} rows)\n")

    log.info("All source CSVs written to FETT/data/source/")


if __name__ == "__main__":
    main()
