from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch as th
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops

try:
    from base.dataset.dataset import ddgDataSet, ddgData
    from base.dataset import utils as ds_utils
except ImportError:
    from dataset import ddgDataSet, ddgData
    from dataset import utils as ds_utils


# ─── Path resolution for minimal (website) manifests ─────────────
# A web user only knows (compound_id, kinase_A, kinase_B). The actual
# pdb/sdf files are derived from where the uploaded structures live
# (`structures_root`) plus a naming convention. Templates use {kinase}
# and {compound_id} placeholders and are RELATIVE to structures_root.
DEFAULT_STRUCTURES_ROOT = "inference_data"
DEFAULT_PROTEIN_TEMPLATE = "{kinase}/{kinase}.pdb"
DEFAULT_LIGAND_TEMPLATE = "{kinase}/{compound_id}.sdf"


def _join_root(structures_root, rel: str) -> str:
    if structures_root in (None, ""):
        return rel
    return str(Path(structures_root) / rel)


def resolve_pair_paths(compound_id, kinase_A, kinase_B,
                       structures_root=DEFAULT_STRUCTURES_ROOT,
                       protein_template=DEFAULT_PROTEIN_TEMPLATE,
                       ligand_template=DEFAULT_LIGAND_TEMPLATE):
    """Build (proteinA_pdb, ligandA_sdf, proteinB_pdb, ligandB_sdf) from the
    minimal identifiers a website user provides. Protein files depend only on
    the kinase; ligand files depend on the (kinase, compound) docked pose."""
    cid = str(compound_id)

    def prot(k):
        return _join_root(structures_root,
                          protein_template.format(kinase=str(k), compound_id=cid))

    def lig(k):
        return _join_root(structures_root,
                          ligand_template.format(kinase=str(k), compound_id=cid))

    return prot(kinase_A), lig(kinase_A), prot(kinase_B), lig(kinase_B)


def _nonempty(v) -> bool:
    """True if a manifest cell holds a real value (not None / NaN / blank)."""
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    if isinstance(v, str) and v.strip() == "":
        return False
    return True


class LiveInferenceDataSet(ddgDataSet):
    """Same featurization as `ddgDataSet`, but reads pocket.pdb + ligand.sdf
    on the fly instead of preprocessed parquet files."""

    def __init__(self,
                 structures_root: str = DEFAULT_STRUCTURES_ROOT,
                 protein_template: str = DEFAULT_PROTEIN_TEMPLATE,
                 ligand_template: str = DEFAULT_LIGAND_TEMPLATE,
                 **kwargs):
        # Pass featurization params (interaction_dist, typing_mode,
        # cache_frames, ...) straight through to the base dataset.
        super().__init__(**kwargs)
        self.structures_root = structures_root
        self.protein_template = protein_template
        self.ligand_template = ligand_template

    def populate(self, input_file: Path, overwrite: bool = True):
        """Load a manifest CSV.

        Minimal (website) form — only these three columns are required::

            compound_id,kinase_A,kinase_B
            99960618,P23458,P52333

        The four structure paths are constructed from `self.structures_root`
        and the naming templates. A `label` column is optional and only used
        for evaluation; predictions never need it.

        Explicit `proteinA_pdb / ligandA_sdf / proteinB_pdb / ligandB_sdf`
        columns are still honored when present (per-cell): any blank cell is
        filled in from the templates, so full and minimal manifests both work.
        """
        inf = pd.read_csv(input_file)

        required = {"compound_id", "kinase_A", "kinase_B"}
        missing = required - set(inf.columns)
        if missing:
            raise ValueError(
                f"Manifest is missing required column(s): {sorted(missing)}. "
                f"At minimum it must contain: compound_id, kinase_A, kinase_B."
            )

        labels = (inf["label"].tolist() if "label" in inf.columns
                  else [float("nan")] * len(inf))

        has = set(inf.columns)
        entries = []
        for _, row in inf.iterrows():
            pA, lA, pB, lB = resolve_pair_paths(
                row["compound_id"], row["kinase_A"], row["kinase_B"],
                self.structures_root, self.protein_template, self.ligand_template,
            )

            def pick(col, default):
                v = row[col] if col in has else None
                return v if _nonempty(v) else default

            entries.append({
                "compound_id":  row["compound_id"],
                "kinase_A":     row["kinase_A"],
                "kinase_B":     row["kinase_B"],
                "proteinA_pdb": pick("proteinA_pdb", pA),
                "ligandA_sdf":  pick("ligandA_sdf",  lA),
                "proteinB_pdb": pick("proteinB_pdb", pB),
                "ligandB_sdf":  pick("ligandB_sdf",  lB),
            })

        if overwrite:
            self.entries = entries
            self.labels = labels
        else:
            self.entries += entries
            self.labels += labels

    # ------------------------------------------------------------------
    # In-memory equivalent of `parse_complex_to_parquet` from utils.py.
    # Mirrors that function exactly, but returns a DataFrame instead of
    # writing parquet. Kept here to avoid changing utils.py.
    # ------------------------------------------------------------------
    @staticmethod
    def _build_complex_df(protein_pdb: str, ligand_sdf: str) -> pd.DataFrame:
        from biopandas.pdb import PandasPdb
        from openbabel import pybel
        try:
            from base.atom_types.atom_types import Typer
        except ImportError:
            from atom_types.atom_types import Typer

        typer = Typer()

        # ---- Protein ----
        prot_df = PandasPdb().read_pdb(str(protein_pdb)).df["ATOM"]
        if "element_symbol" in prot_df.columns:
            prot_df = prot_df[prot_df["element_symbol"].str.strip() != "H"
                              ].reset_index(drop=True)
        else:
            prot_df = prot_df[~prot_df["atom_name"].str.strip()
                              .str.match(r"^(\d*H|H)")].reset_index(drop=True)

        prot_df["is_ligand"] = 0
        prot_df["formal_charge"] = 0

        try:
            prot_types, prot_occ = typer.run(protein_pdb)
            if len(prot_types) != len(prot_df):
                min_len = min(len(prot_types), len(prot_df))
                prot_types = prot_types[:min_len]
                prot_occ = prot_occ[:min_len]
                prot_df = prot_df.iloc[:min_len].reset_index(drop=True)
        except Exception as e:
            print(f"[WARN] Protein typing failed for {protein_pdb}: {e}")
            prot_types = [0] * len(prot_df)
            prot_occ = [1] * len(prot_df)

        prot_df["lmg_types"] = prot_types
        prot_df["occ"] = prot_occ

        # ---- Ligand ----
        mol = list(pybel.readfile("sdf", str(ligand_sdf)))[0]
        lig_rows = []
        for atom in mol:
            if atom.atomicnum == 1:
                continue
            smina_type = typer.obatom_to_smina_type(atom)
            if smina_type == "NumTypes":
                smina_type_int = len(typer.atom_type_data)
            else:
                smina_type_int = typer.atom_types.index(smina_type)
            lig_rows.append({
                "atom_number":   atom.idx + 1,
                "atom_name":     atom.residue.OBResidue.GetAtomID(atom.OBAtom).strip(),
                "chain_id":      "L",
                "residue_number": 1,
                "insertion":     "",
                "x_coord":       atom.coords[0],
                "y_coord":       atom.coords[1],
                "z_coord":       atom.coords[2],
                "residue_name":  "LIG",
                "is_ligand":     1,
                "lmg_types":     smina_type_int,
                "occ":           1,
                "formal_charge": atom.formalcharge,
            })
        lig_df = pd.DataFrame(lig_rows)

        keep_cols = ["atom_number", "atom_name", "chain_id", "residue_number",
                     "insertion", "x_coord", "y_coord", "z_coord",
                     "residue_name", "is_ligand", "lmg_types", "occ",
                     "formal_charge"]
        return pd.concat([prot_df[keep_cols], lig_df[keep_cols]],
                         ignore_index=True)

    # ------------------------------------------------------------------
    # Live equivalent of `_load_and_build_graph` — no parquet, no cache.
    # ------------------------------------------------------------------
    def _build_one_graph(self, protein_pdb: str, ligand_sdf: str,
                        label, graph_id: str) -> Data:
        cache_key = (protein_pdb, ligand_sdf)
        if self.cache_frames and cache_key in self.cache:
            complex_df = self.cache[cache_key].copy()
        else:
            complex_df = self._build_complex_df(protein_pdb, ligand_sdf)
            if self.cache_frames:
                self.cache[cache_key] = complex_df.copy()

        prot_df = complex_df[complex_df["is_ligand"] == 0].reset_index(drop=True)
        lig_df  = complex_df[complex_df["is_ligand"] == 1].reset_index(drop=True)

        result = self._build_graph(prot_df, lig_df)
        if result is None:
            return None

        feats, edge_indices, edge_attr = result

        edge_index, edge_attr_tensor = remove_self_loops(
            edge_index=th.from_numpy(edge_indices).long(),
            edge_attr=th.from_numpy(edge_attr),
        )

        # label may be NaN if the manifest had no `label` column — that's
        # fine, pred-only outputs don't need it. We just can't use it for
        # training-style metrics.
        y_val = float(label) if label is not None and not (
            isinstance(label, float) and math.isnan(label)) else 0.0

        return Data(
            x=th.from_numpy(feats[:, 3:]).float(),
            edge_index=edge_index,
            edge_attr=edge_attr_tensor.float(),
            pos=th.from_numpy(feats[:, :3]).float(),
            y=th.tensor(y_val).float(),
            # `pdb_file` is what ddgData copies into `pdb_wt` / `pdb_mut`
            # and what the inference CSV writer keys predictions on. We
            # stuff the ligand path in too so each row is uniquely
            # identifiable downstream.
            pdb_file=f"{protein_pdb}|{ligand_sdf}",
            wt_mut=graph_id,
        )

    def __getitem__(self, idx: int):
        label = self.labels[idx]
        entry = self.entries[idx]

        graph_A = self._build_one_graph(
            entry["proteinA_pdb"], entry["ligandA_sdf"], label, graph_id="A",
        )
        if graph_A is None:
            print(f"[WARN] idx={idx} side-A graph build failed, skipping to next")
            return self.__getitem__((idx + 1) % len(self))

        graph_B = self._build_one_graph(
            entry["proteinB_pdb"], entry["ligandB_sdf"], label, graph_id="B",
        )
        if graph_B is None:
            print(f"[WARN] idx={idx} side-B graph build failed, skipping to next")
            return self.__getitem__((idx + 1) % len(self))

        graph = self.__aggregate_graphs__([graph_A, graph_B])
        # Attach manifest-level identifiers so the inference writer can
        # produce a CSV keyed by compound + kinases rather than file paths.
        graph.compound_id = str(entry["compound_id"])
        graph.kinase_A    = str(entry["kinase_A"])
        graph.kinase_B    = str(entry["kinase_B"])
        return graph