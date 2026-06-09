import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from biopandas.pdb import PandasPdb
import numpy as np
import pandas as pd

try:
    from base.atom_types.atom_types import Typer
except:
    from atom_types.atom_types import Typer


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def get_type_map(types: list=None):
    t = Typer()

    if types is None:
        types = [
            ['AliphaticCarbonXSHydrophobe'],
            ['AliphaticCarbonXSNonHydrophobe'],
            ['AromaticCarbonXSHydrophobe'],
            ['AromaticCarbonXSNonHydrophobe'],
            ['Nitrogen', 'NitrogenXSAcceptor'],
            ['NitrogenXSDonor', 'NitrogenXSDonorAcceptor'],
            ['Oxygen', 'OxygenXSAcceptor'],
            ['OxygenXSDonor', 'OxygenXSDonorAcceptor'],
            ['Sulfur', 'SulfurAcceptor'],
            ['Phosphorus']
        ]
    out_dict = {}
    generic = []
    for i, element_name in enumerate(t.atom_types):
        for types_list in types:
            if element_name in types_list:
                out_dict[i] = types.index(types_list)
                break
        if not i in out_dict.keys():
            generic.append(i)

    generic_type = len(types)
    for other_type in generic:
        out_dict[other_type] = generic_type
    return out_dict


def parse_pdb_to_parquet(pdb_file: Path, parquet_path: Path, lmg_typed: bool = True, ca: bool = False):
    """Parses a pdb file to smaller, faster parquet df format. Returns the resulting df.
    Args:
        pdb_file (Path): Path to the pdb file
        parquet_path (Path): Output filename
        lmg_typed (bool, optional): Use typer functionality to generate lmg types for each atom. 
            Defaults to True.
    Returns:
        pd.DataFrame: The pdb df.
    """
    pdb_df = PandasPdb().read_pdb(str(pdb_file)).df["ATOM"]

    # remove individually resolved hydrogens (not present in most files, ie add noise)
    bool_sel = pdb_df['atom_name'].apply(lambda x: x.strip() not in ['H'])
    pdb_df = pdb_df[bool_sel].reset_index(drop=True)

    # store lmg typings - types and occupancies
    if lmg_typed:
        typer = Typer()
        types, occupancies = typer.run(pdb_file)

        # Handle length mismatch between OpenBabel and BioPandas parsing
        if len(types) != len(pdb_df):
            diff = abs(len(types) - len(pdb_df))
            print(f"[WARN] Atom count mismatch in {pdb_file}: "
                  f"OpenBabel={len(types)}, BioPandas={len(pdb_df)}, diff={diff}")

            if diff <= 5:
                # Small mismatch — truncate to shorter length
                min_len = min(len(types), len(pdb_df))
                types = types[:min_len]
                occupancies = occupancies[:min_len]
                pdb_df = pdb_df.iloc[:min_len].reset_index(drop=True)
            else:
                # Large mismatch — file is too corrupted
                raise ValueError(
                    f"Atom count mismatch too large ({diff}) in {pdb_file}: "
                    f"OpenBabel={len(types)}, BioPandas={len(pdb_df)}"
                )

        pdb_df['lmg_types'] = types
        pdb_df['occ'] = occupancies

    # drop columns that are not needed downstream
    pdb_df = pdb_df[['atom_number', 'atom_name', 'chain_id', 'residue_number', 'insertion',
                      'x_coord', 'y_coord', 'z_coord', 'occ', 'lmg_types', 'residue_name']]

    if ca:
        pdb_df = pdb_df[pdb_df.atom_name == 'CA']

    pdb_df.to_parquet(parquet_path)
    return pdb_df

from openbabel import pybel
def parse_complex_to_parquet(protein_pdb: Path, ligand_sdf: Path, parquet_path: Path,
                              lmg_typed: bool = True):
    """Parse protein (PDB) + ligand (SDF) into a single parquet dataframe."""

    typer = Typer()

    # ---- Protein ----
    prot_df = PandasPdb().read_pdb(str(protein_pdb)).df["ATOM"]
    #prot_df = prot_df[prot_df['atom_name'].apply(lambda x: x.strip() != 'H')].reset_index(drop=True)
    # Filter hydrogens - check element_symbol column (more reliable)
    if 'element_symbol' in prot_df.columns:
        prot_df = prot_df[prot_df['element_symbol'].str.strip() != 'H'].reset_index(drop=True)
    else:
        # Fallback: filter by atom_name pattern (H, HA, HB1, 1H, 2HG, etc.)
        prot_df = prot_df[~prot_df['atom_name'].str.strip().str.match(r'^(\d*H|H)')].reset_index(drop=True)
   
    prot_df['is_ligand'] = 0
    prot_df['formal_charge'] = 0  # Proteins: assume neutral for heavy atoms

    if lmg_typed:
        try:
            prot_types, prot_occ = typer.run(protein_pdb)
            if len(prot_types) != len(prot_df):
                min_len = min(len(prot_types), len(prot_df))
                prot_types = prot_types[:min_len]
                prot_occ = prot_occ[:min_len]
                prot_df = prot_df.iloc[:min_len].reset_index(drop=True)
        except Exception as e:
            print(f"[WARN] Protein typing failed: {e}")
            prot_types = [0] * len(prot_df)
            prot_occ = [1] * len(prot_df)

        prot_df['lmg_types'] = prot_types
        prot_df['occ'] = prot_occ

    # ---- Ligand (from SDF) ----
    mol = list(pybel.readfile("sdf", str(ligand_sdf)))[0]

    lig_rows = []
    for atom in mol:
        # Skip hydrogens
        if atom.atomicnum == 1:
            continue

        smina_type = typer.obatom_to_smina_type(atom)
        if smina_type == "NumTypes":
            smina_type_int = len(typer.atom_type_data)
        else:
            smina_type_int = typer.atom_types.index(smina_type)

        # Get formal charge from OpenBabel
        formal_charge = atom.formalcharge  # e.g., -1 for COO-, +1 for quaternary N

        lig_rows.append({
            'atom_number': atom.idx + 1,
            'atom_name': atom.residue.OBResidue.GetAtomID(atom.OBAtom).strip(),
            'chain_id': 'L',
            'residue_number': 1,
            'insertion': '',
            'x_coord': atom.coords[0],
            'y_coord': atom.coords[1],
            'z_coord': atom.coords[2],
            'residue_name': 'LIG',
            'is_ligand': 1,
            'lmg_types': smina_type_int,
            'occ': 1,
            'formal_charge': formal_charge,
        })

    lig_df = pd.DataFrame(lig_rows)
    print(f"[INFO] Protein atoms: {len(prot_df)}, Ligand atoms: {len(lig_df)}")
    
    # Log charged atoms for debugging
    charged_atoms = lig_df[lig_df['formal_charge'] != 0]
    if len(charged_atoms) > 0:
        print(f"[INFO] Charged ligand atoms: {len(charged_atoms)}")
        print(charged_atoms[['atom_name', 'formal_charge']].to_string())

    # ---- Combine ----
    keep_cols = ['atom_number', 'atom_name', 'chain_id', 'residue_number', 'insertion',
                 'x_coord', 'y_coord', 'z_coord', 'residue_name', 'is_ligand', 
                 'lmg_types', 'occ', 'formal_charge']
    prot_df = prot_df[keep_cols]
    lig_df = lig_df[keep_cols]

    complex_df = pd.concat([prot_df, lig_df], ignore_index=True)
    complex_df.to_parquet(parquet_path)
    return complex_df