"""
Query the Materials Project API for all entries with band gaps per DFT functional.
Also saves an mp-id → formation energy lookup dictionary used by 03_combine_raw.py.

Run once from project root:
    MP_API_KEY=<your_key> uv run python FETT/scripts/data/02_query_mp.py

Outputs to FETT/data/interim/:
    mp_GGA.csv         (mpid, formula, BG, FE, e_above_hull)  — GGA + PBE run_types
    mp_GGAU.csv        (mpid, formula, BG, FE, e_above_hull)  — GGA+U
    mp_PBESOL.csv      (mpid, formula, BG, FE, e_above_hull)  — PBEsol
    mp_SCAN.csv        (mpid, formula, BG, FE, e_above_hull)  — SCAN + r2SCAN
    mp_HSE.csv         (mpid, formula, BG, FE, e_above_hull)  — HSE06 (may be empty)
    mp_fe_lookup.json  (dict: mpid → formation_energy_per_atom)

Deduplication:
    Per material_id: lowest e_above_hull task is kept (ground state for that mp-id).
    Per formula: lowest formation energy entry is kept across all mp-ids.
"""

import json
import logging
import os
import warnings
from pathlib import Path

import pandas as pd
from mp_api.client import MPRester

# Suppress urllib3 "Connection pool is full" warnings — these are benign.
# mp_api fetches chunks from S3 in parallel; urllib3 discards idle connections
# when the pool fills up but all downloads still complete successfully.
warnings.filterwarnings("ignore", message="Connection pool is full")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
SOURCE_OUT_DIR = SCRIPT_DIR.parent.parent / "data" / "interim"
SOURCE_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Maps MP run_type strings → output file key.
# "PBE" is MP's alias for the same GGA functional; "PBEsol" is a distinct functional.
# r2SCAN is included under SCAN (it's a numerically-improved variant of SCAN).
# Multiple capitalizations are listed defensively — MP API has used both "r2SCAN" and "R2SCAN".
FUNCTIONAL_KEY = {
    "GGA": "GGA",
    "PBE": "GGA",       # alias used by some MP tasks
    "GGA+U": "GGAU",
    "PBEsol": "PBESOL",
    "SCAN": "SCAN",
    "r2SCAN": "SCAN",   # regularized-restored SCAN → grouped with SCAN
    "R2SCAN": "SCAN",   # alternate capitalization seen in some API responses
    "r2scan": "SCAN",   # lowercase variant
    "HSE06": "HSE",
}


def main():
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MP_API_KEY environment variable not set.\n"
            "Get a key at https://materialsproject.org/api and run:\n"
            "    MP_API_KEY=<your_key> uv run python FETT/scripts/data/02_query_mp.py"
        )

    # chunk_size limits how many results are fetched per HTTP request, reducing
    # the risk of a single request timing out on large datasets.
    CHUNK_SIZE = 1000

    with MPRester(api_key) as mpr:
        # 1. Summary: formation energy and e_above_hull per material_id
        log.info("Querying summary (formation energies) for all materials ...")
        summary_docs = mpr.materials.summary.search(
            fields=["material_id", "formula_pretty", "formation_energy_per_atom", "energy_above_hull"],
            chunk_size=CHUNK_SIZE,
        )
        log.info(f"  {len(summary_docs):,} materials in summary")

        fe_by_mpid: dict[str, float | None] = {}
        ehull_by_mpid: dict[str, float | None] = {}
        formula_by_mpid: dict[str, str] = {}
        for doc in summary_docs:
            mpid = str(doc.material_id)
            fe_by_mpid[mpid] = doc.formation_energy_per_atom
            ehull_by_mpid[mpid] = doc.energy_above_hull
            formula_by_mpid[mpid] = doc.formula_pretty

        # 2. Materials: task_id → run_type mapping
        log.info("Querying materials for task_id → functional mapping ...")
        materials_docs = mpr.materials.search(
            fields=["material_id", "task_ids", "run_types"],
            chunk_size=CHUNK_SIZE,
        )
        log.info(f"  {len(materials_docs):,} materials with task info")

        task_to_mpid: dict[str, str] = {}
        task_to_functional: dict[str, str] = {}
        for doc in materials_docs:
            mpid = str(doc.material_id)
            run_types = doc.run_types or {}
            for task_id in doc.task_ids or []:
                tid = str(task_id)
                task_to_mpid[tid] = mpid
                rt = run_types.get(task_id)
                if rt is not None:
                    task_to_functional[tid] = rt.value if hasattr(rt, "value") else str(rt)

        # 3. Electronic structure: band gap per task
        log.info("Querying electronic structure for all band gaps ...")
        es_docs = mpr.materials.electronic_structure.search(
            fields=["material_id", "task_id", "band_gap"],
            chunk_size=CHUNK_SIZE,
        )
        log.info(f"  {len(es_docs):,} electronic structure tasks")

        # 4. Assemble per-functional rows
        rows_by_key: dict[str, list] = {k: [] for k in set(FUNCTIONAL_KEY.values())}
        skipped = 0
        for doc in es_docs:
            mpid = str(doc.material_id)
            tid = str(doc.task_id)
            raw_functional = task_to_functional.get(tid, "unknown")
            key = FUNCTIONAL_KEY.get(raw_functional)
            if key is None:
                skipped += 1
                continue
            rows_by_key[key].append({
                "mpid": mpid,
                "formula": formula_by_mpid.get(mpid, ""),
                "BG": doc.band_gap,
                "FE": fe_by_mpid.get(mpid),
                "e_above_hull": ehull_by_mpid.get(mpid),
            })

        if skipped:
            log.warning(f"  {skipped:,} tasks skipped (unmapped functional: {set(task_to_functional.values()) - set(FUNCTIONAL_KEY)})")

    # 5. Save per-functional CSVs
    for key, rows in rows_by_key.items():
        if not rows:
            log.warning(f"  No data collected for functional key '{key}', skipping")
            continue
        df = pd.DataFrame(rows)
        df = df.dropna(subset=["formula"])

        # Keep the lowest e_above_hull task per material_id (ground state structure),
        # then keep the lowest FE entry per formula (most stable composition polymorph).
        df = (
            df.sort_values("e_above_hull", na_position="last")
            .drop_duplicates(subset=["mpid"])
            .sort_values("FE", na_position="last")
            .drop_duplicates(subset=["formula"])
            .reset_index(drop=True)
        )
        out = SOURCE_OUT_DIR / f"mp_{key}.csv"
        df.to_csv(out, index=False)
        log.info(f"Saved {out.name} ({len(df):,} rows)")

    # 6. Save FE lookup (all mp-ids, used by 03_combine_raw.py for sources without FE columns)
    fe_lookup = {mpid: fe for mpid, fe in fe_by_mpid.items() if fe is not None}
    out = SOURCE_OUT_DIR / "mp_fe_lookup.json"
    with open(out, "w") as f:
        json.dump(fe_lookup, f)
    log.info(f"Saved {out.name} ({len(fe_lookup):,} entries)")

    log.info("MP query complete!")


if __name__ == "__main__":
    main()
