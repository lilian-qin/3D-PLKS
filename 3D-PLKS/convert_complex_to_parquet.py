import pandas as pd
from pathlib import Path
import sys
import os
from multiprocessing import Pool, cpu_count

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from base import utils


def convert_one(args):
    """Convert a single protein-ligand complex to parquet."""
    pro_path, lig_path, parquet_path = args

    if Path(parquet_path).exists():
        return f"SKIP: {parquet_path}"

    try:
        Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
        utils.parse_complex_to_parquet(pro_path, lig_path, parquet_path, lmg_typed=True)
        return f"OK: {parquet_path}"
    except Exception as e:
        return f"FAIL: {parquet_path} — {e}"


def discover_complexes(docking_dir: Path, output_dir: Path):
    """
    Scan docking folder structure and build task list.
    
    Structure expected:
        docking_dir/{uniprot_id}/{uniprot_id}.pdb
        docking_dir/{uniprot_id}/*.sdf  (one or more ligands)
    
    Output:
        output_dir/{uniprot_id}/{uniprot_id}_{ligand_name}.parquet
    """
    tasks = []
    
    for uniprot_folder in sorted(docking_dir.iterdir()):
        if not uniprot_folder.is_dir():
            continue
        
        uniprot_id = uniprot_folder.name
        protein_pdb = uniprot_folder / f"{uniprot_id}.pdb"
        
        if not protein_pdb.exists():
            print(f"[WARN] No protein PDB found: {protein_pdb}")
            continue
        
        # Find all ligand SDF files in this folder
        sdf_files = list(uniprot_folder.glob("*.sdf"))
        
        if not sdf_files:
            print(f"[WARN] No SDF files in {uniprot_folder}")
            continue
        
        for lig_sdf in sdf_files:
            lig_name = lig_sdf.stem  # e.g., "CHEMBL12345" from "CHEMBL12345.sdf"
            
            # Output: ../data_process/all_egnngraph/{uniprot_id}/{uniprot_id}_{lig_name}.parquet
            parquet_path = output_dir / uniprot_id / f"{uniprot_id}_{lig_name}.parquet"
            
            tasks.append((str(protein_pdb), str(lig_sdf), str(parquet_path)))
    
    return tasks


def main():
    docking_dir = Path("../data_process/cases_data/docking/")
    output_dir = Path("../data_process/cases_data/egnngraph")
    num_workers = 4

    print(f"Scanning docking folder: {docking_dir.resolve()}")
    print(f"Output folder: {output_dir.resolve()}\n")

    tasks = discover_complexes(docking_dir, output_dir)
    
    print(f"Total complexes to convert: {len(tasks)}")
    print(f"Using {num_workers} workers\n")

    if not tasks:
        print("No tasks found. Check folder structure.")
        return

    # Run
    ok, skip, miss, fail = 0, 0, 0, 0
    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(convert_one, tasks)):
            if result.startswith("OK"):
                ok += 1
            elif result.startswith("SKIP"):
                skip += 1
            elif result.startswith("MISS"):
                miss += 1
                print(result)
            else:
                fail += 1
                print(result)

            if (i + 1) % 500 == 0:
                print(f"Progress: {i+1}/{len(tasks)} | OK={ok} SKIP={skip} MISS={miss} FAIL={fail}")

    print(f"\nDone! OK={ok} SKIP={skip} MISS={miss} FAIL={fail}")


if __name__ == "__main__":
    main()