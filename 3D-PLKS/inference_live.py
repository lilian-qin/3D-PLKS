import argparse
import csv
import re
import yaml
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from collections import defaultdict, OrderedDict
from torch_geometric.loader import DataLoader

from models.egnn_model import ddgEGNN
from base.dataset_live import (
    LiveInferenceDataSet,
    resolve_pair_paths,
    DEFAULT_STRUCTURES_ROOT,
    DEFAULT_PROTEIN_TEMPLATE,
    DEFAULT_LIGAND_TEMPLATE,
)


_PEARSON_RE = re.compile(r"val_pearson=([\-0-9.]+)")
_SEED_DIR_RE = re.compile(r"^seed_(\d+)$")


# ─── Checkpoint discovery (unchanged) ─────────────────────────────

def parse_pearson_from_name(p: Path) -> float:
    m = _PEARSON_RE.search(p.name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return float("-inf")


def find_best_checkpoint(ckpt_dir: Path) -> tuple[Path, float]:
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
    best = max(ckpts, key=parse_pearson_from_name)
    return best, parse_pearson_from_name(best)


def discover_seed_dirs(parent: Path) -> list[Path]:
    if not parent.is_dir():
        raise FileNotFoundError(f"Parent dir not found: {parent}")
    out = []
    for child in sorted(parent.iterdir(),
                        key=lambda p: int(_SEED_DIR_RE.match(p.name).group(1))
                        if _SEED_DIR_RE.match(p.name) else 1 << 30):
        if not child.is_dir():
            continue
        if not _SEED_DIR_RE.match(child.name):
            continue
        if not any(child.glob("*.ckpt")):
            print(f"[warn] {child} matches seed_* but contains no .ckpt — skipping")
            continue
        out.append(child)
    if not out:
        raise FileNotFoundError(
            f"No 'seed_<int>/' subdirectories with .ckpt files under {parent}"
        )
    return out


# ─── Manifest pre-screen for missing dock files ──────────────────

def _nonempty(v) -> bool:
    return isinstance(v, str) and v.strip() != ""


def screen_manifest_for_missing(manifest_path: str,
                                present_out: str,
                                failed_out: str,
                                structures_root: str = DEFAULT_STRUCTURES_ROOT,
                                protein_template: str = DEFAULT_PROTEIN_TEMPLATE,
                                ligand_template: str = DEFAULT_LIGAND_TEMPLATE
                                ) -> tuple[str, int, int]:
    """Split a manifest into rows whose four structure files all exist and
    rows with at least one missing file.

    Works on both the minimal website manifest (compound_id, kinase_A,
    kinase_B) and full manifests with explicit path columns: for each row the
    four paths are taken from explicit columns when present and non-blank,
    otherwise constructed from `structures_root` + templates — exactly the
    same resolution the dataset uses, so the existence check matches what the
    loader will actually open.

    - Rows with all files present -> `present_out` (used for inference); the
      original columns are preserved unchanged.
    - Rows with >=1 missing file  -> `failed_out`, with an added
      `missing_files` column naming the absent file(s). For a website this
      doubles as the "you still need to upload these structures" report.

    Paths are resolved relative to the current working directory, matching how
    the dataloader (pybel.readfile on the raw string) resolves them.

    Returns (present_out, n_present, n_failed).
    """
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    req = {"compound_id", "kinase_A", "kinase_B"}
    miss_cols = req - set(fieldnames)
    if miss_cols:
        raise ValueError(
            f"Manifest is missing required column(s): {sorted(miss_cols)}. "
            f"At minimum it must contain: compound_id, kinase_A, kinase_B."
        )

    exist_cache: dict[str, bool] = {}
    def _exists(p: str) -> bool:
        if p not in exist_cache:
            exist_cache[p] = Path(p).is_file()
        return exist_cache[p]

    present, failed = [], []
    for row in rows:
        pA, lA, pB, lB = resolve_pair_paths(
            row["compound_id"], row["kinase_A"], row["kinase_B"],
            structures_root, protein_template, ligand_template,
        )
        paths = {
            "proteinA_pdb": row.get("proteinA_pdb") if _nonempty(row.get("proteinA_pdb")) else pA,
            "ligandA_sdf":  row.get("ligandA_sdf")  if _nonempty(row.get("ligandA_sdf"))  else lA,
            "proteinB_pdb": row.get("proteinB_pdb") if _nonempty(row.get("proteinB_pdb")) else pB,
            "ligandB_sdf":  row.get("ligandB_sdf")  if _nonempty(row.get("ligandB_sdf"))  else lB,
        }
        missing = [p for p in paths.values() if not _exists(p)]
        if missing:
            r = dict(row)
            r["missing_files"] = ";".join(missing)
            failed.append(r)
        else:
            present.append(row)

    Path(present_out).parent.mkdir(parents=True, exist_ok=True)
    with open(present_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(present)

    if failed:
        Path(failed_out).parent.mkdir(parents=True, exist_ok=True)
        with open(failed_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fieldnames) + ["missing_files"])
            w.writeheader()
            w.writerows(failed)

    print(f"[screen] {manifest_path}: {len(present)} present pair(s), "
          f"{len(failed)} pair(s) with missing file(s)")
    if failed:
        distinct = sorted({p for r in failed
                           for p in r["missing_files"].split(";")})
        by_folder: dict[str, int] = {}
        for p in distinct:
            folder = Path(p).parent.name
            by_folder[folder] = by_folder.get(folder, 0) + 1
        print(f"[screen] {len(distinct)} distinct missing file(s) by folder:")
        for folder, cnt in sorted(by_folder.items(), key=lambda x: -x[1]):
            print(f"           {folder}: {cnt}")
        print(f"[screen] skipped pairs collected in: {failed_out}")
    return present_out, len(present), len(failed)


# ─── Inference plumbing ──────────────────────────────────────────

def build_trainer(trainer_config: dict) -> pl.Trainer:
    tc = (trainer_config or {}).copy()
    for k in ("resume_from_checkpoint", "gpus", "auto_select_gpus",
              "max_epochs", "min_epochs", "max_steps", "min_steps",
              "accumulate_grad_batches", "val_check_interval",
              "deterministic"):
        tc.pop(k, None)
    tc.setdefault("accelerator", "gpu" if torch.cuda.is_available() else "cpu")
    tc.setdefault("devices", 1)
    return pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        **tc,
    )


def build_live_loader(manifest_path: str, config: dict) -> DataLoader:
    """Build a DataLoader from a manifest CSV (pdb + sdf rows)."""
    ds_params = (config["dataset_params"] or {}).copy()
    # Strip params LiveInferenceDataSet doesn't know about (paths to
    # precomputed dirs, parquet roots, etc). Keep the featurization params
    # plus the structure-path resolution params used for minimal manifests.
    init_kwargs = {
        k: ds_params[k] for k in
        ("interaction_dist", "typing_mode", "cache_frames",
         "structures_root", "protein_template", "ligand_template")
        if k in ds_params
    }
    ds = LiveInferenceDataSet(**init_kwargs)
    ds.populate(manifest_path, overwrite=True)
    print(f"[data] Manifest: {manifest_path} — {len(ds)} pair(s)")

    loader_params = (config["loader_params"] or {}).copy()
    return DataLoader(
        ds,
        batch_size=loader_params.get("batch_size", 1),
        shuffle=False,
        num_workers=loader_params.get("num_workers", 0),
        follow_batch=["x_0", "x_1"],
    )


def _make_manifest_test_step(model):
    """Build a replacement test_step that records compound_id / kinase_A /
    kinase_B (attached by LiveInferenceDataSet) instead of pdb file paths."""
    import types

    def test_step(self, batch, batch_idx):
        pred = self.forward(batch)
        pred_np = pred.flatten().detach().cpu().numpy()

        # Labels may be NaN (manifest had no `label` col); record as-is.
        y = batch.y
        labels = y.detach().cpu().numpy().flatten() if y is not None \
                 else [float("nan")] * len(pred_np)

        # ddgData batches list-valued attrs into Python lists, so these
        # come through as lists of length = batch size.
        cids   = batch.compound_id if isinstance(batch.compound_id, list) \
                 else [batch.compound_id]
        kAs    = batch.kinase_A    if isinstance(batch.kinase_A, list) \
                 else [batch.kinase_A]
        kBs    = batch.kinase_B    if isinstance(batch.kinase_B, list) \
                 else [batch.kinase_B]

        for i, score in enumerate(pred_np):
            self.test_set_predictions.append(
                (cids[i], kAs[i], kBs[i], float(score), float(labels[i]))
            )
        return {"pred": pred, "y": y}

    model.test_step = types.MethodType(test_step, model)


def load_and_infer(ckpt_path: Path, config: dict,
                   manifest_path: str | None) -> list:
    """Load one checkpoint, run test, return predictions.

    Manifest mode returns: list of (compound_id, kinase_A, kinase_B, pred, label)
    Precomputed mode returns: list of (pdb_wt, pdb_mut, pred, label)
    """
    print(f"[model] Loading: {ckpt_path}")
    model = ddgEGNN.load_from_checkpoint(
        str(ckpt_path),
        dataset_config=config["dataset_params"],
        loader_config=config["loader_params"],
        trainer_config=config["trainer_params"],
        **(config["model_params"] or {}),
    )
    model.eval()
    model.test_set_predictions = []

    # Inference-only: skip epoch-end metrics. They're not needed (we just
    # want predictions) and scipy's pearsonr/spearmanr require n >= 2,
    # which would crash on small manifests.
    model.on_test_epoch_end = lambda: None

    trainer = build_trainer(config["trainer_params"] or {})

    if manifest_path is not None:
        # Manifest mode: override test_step so output rows are keyed by
        # (compound_id, kinase_A, kinase_B) instead of file paths, then
        # drive the model with our own DataLoader.
        _make_manifest_test_step(model)
        loader = build_live_loader(manifest_path, config)
        trainer.test(model, dataloaders=loader, ckpt_path=None)
    else:
        # Original behavior: model defines its own test_dataloader from
        # the config's precomputed test dir.
        trainer.test(model, ckpt_path=None)

    preds = list(model.test_set_predictions)

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return preds


# ─── Main ────────────────────────────────────────────────────────

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = defaultdict(lambda: None, config)

    # ── Input resolution: manifest vs precomputed test dir ──────
    if args.manifest:
        mp = Path(args.manifest)
        if not mp.is_file():
            raise FileNotFoundError(f"--manifest not found: {mp}")

        # Resolve structure-path settings: CLI overrides config, config
        # overrides built-in defaults. Push them back into dataset_params so
        # the dataset (build_live_loader) resolves paths identically.
        ds_params = config["dataset_params"] or {}
        structures_root = (args.structures_root
                           or ds_params.get("structures_root")
                           or DEFAULT_STRUCTURES_ROOT)
        protein_template = (args.protein_template
                            or ds_params.get("protein_template")
                            or DEFAULT_PROTEIN_TEMPLATE)
        ligand_template = (args.ligand_template
                           or ds_params.get("ligand_template")
                           or DEFAULT_LIGAND_TEMPLATE)
        if config["dataset_params"] is None:
            config["dataset_params"] = {}
        config["dataset_params"]["structures_root"] = structures_root
        config["dataset_params"]["protein_template"] = protein_template
        config["dataset_params"]["ligand_template"] = ligand_template

        # Checkpoint: screen the manifest first so pairs whose pdb/sdf files
        # are missing are skipped (and recorded) instead of crashing the run.
        out_path = Path(args.output)
        failed_out = (args.failed_output
                      or str(out_path.with_name(out_path.stem + "_failed_pairs.csv")))
        present_out = str(out_path.with_name(mp.stem + "_present.csv"))
        present_out, n_present, n_failed = screen_manifest_for_missing(
            str(mp), present_out, failed_out,
            structures_root, protein_template, ligand_template)
        if n_present == 0:
            raise RuntimeError(
                "Every pair in the manifest references at least one missing "
                f"file — nothing to infer. See {failed_out}."
            )
        manifest_path = present_out
        print(f"[data] Live inference from manifest: {manifest_path} "
              f"({n_present} pair(s) to run, {n_failed} skipped)")
    else:
        manifest_path = None
        if args.test_dir:
            if config["dataset_params"] is None:
                raise ValueError("Config has no dataset_params block.")
            config["dataset_params"].setdefault("precomputed_dirs", {})
            config["dataset_params"]["precomputed_dirs"]["test"] = args.test_dir
            print(f"[data] Test dir (overridden): {args.test_dir}")
        else:
            test_dir = (config["dataset_params"] or {}).get(
                "precomputed_dirs", {}).get("test")
            if not test_dir:
                raise ValueError(
                    "No test directory in config and none passed via "
                    "--test_dir (and no --manifest given)."
                )
            print(f"[data] Test dir (from config): {test_dir}")

        test_dir_path = Path(config["dataset_params"]["precomputed_dirs"]["test"])
        if not test_dir_path.is_dir():
            raise FileNotFoundError(f"Test directory does not exist: {test_dir_path}")

    # ── Resolve which checkpoints to use ────────────────────────
    ckpts: list[tuple[Path, float, str]] = []

    if args.ckpt:
        cp = Path(args.ckpt)
        if not cp.is_file():
            raise FileNotFoundError(f"--ckpt not found: {cp}")
        ckpts.append((cp, parse_pearson_from_name(cp), cp.parent.name))

    elif args.seed_dir:
        sd = Path(args.seed_dir)
        best, score = find_best_checkpoint(sd)
        ckpts.append((best, score, sd.name))
        print(f"[ckpt] {sd.name}: best {best.name} (val_pearson={score:.4f})")

    elif args.seed_dirs:
        for d in args.seed_dirs:
            sd = Path(d)
            best, score = find_best_checkpoint(sd)
            ckpts.append((best, score, sd.name))
            print(f"[ckpt] {sd.name}: best {best.name} (val_pearson={score:.4f})")

    elif args.ensemble:
        parent = Path(args.parent_dir) if args.parent_dir \
            else Path(config["save_dir"] or "./")
        print(f"[ckpt] Searching for seed_* dirs under: {parent}")
        for sd in discover_seed_dirs(parent):
            best, score = find_best_checkpoint(sd)
            ckpts.append((best, score, sd.name))
            print(f"[ckpt] {sd.name}: best {best.name} (val_pearson={score:.4f})")

    else:
        raise ValueError(
            "Specify one of: --ckpt, --seed_dir, --seed_dirs, or --ensemble."
        )

    if args.pick_best and len(ckpts) > 1:
        best_i = max(range(len(ckpts)), key=lambda i: ckpts[i][1])
        print(f"\n[ckpt] --pick_best: keeping {ckpts[best_i][2]} "
              f"(val_pearson={ckpts[best_i][1]:.4f})")
        ckpts = [ckpts[best_i]]

    # ── Run inference for each, accumulate per-pair preds ───────
    # Manifest mode rows: (compound_id, kinase_A, kinase_B, pred, label)
    # Precomputed mode rows: (pdb_wt, pdb_mut, pred, label)
    # We key on a tuple of (id_A, id_B [, id_C]) so seeds align even if
    # loader order varies, and remember the "header" for the CSV writer.
    is_manifest = manifest_path is not None
    key_cols = (["compound_id", "kinase_A", "kinase_B"] if is_manifest
                else ["pair_id_A", "pair_id_B"])

    accum: "OrderedDict[tuple, list[float]]" = OrderedDict()
    label_lookup: dict[tuple, float] = {}

    for i, (ckpt_path, _score, tag) in enumerate(ckpts, 1):
        print(f"\n[infer] ({i}/{len(ckpts)}) {tag} — {ckpt_path.name}")
        preds = load_and_infer(ckpt_path, config, manifest_path)
        if not preds:
            raise RuntimeError(f"No predictions from {ckpt_path}")
        print(f"[infer] Got {len(preds)} predictions")

        for row in preds:
            *id_fields, pred, label = row
            key = tuple(id_fields)
            accum.setdefault(key, []).append(float(pred))
            if key not in label_lookup:
                label_lookup[key] = (float(label) if label is not None
                                     else float("nan"))

    n_ckpt = len(ckpts)
    incomplete = [k for k, v in accum.items() if len(v) != n_ckpt]
    if incomplete:
        print(f"\n[warn] {len(incomplete)} pair(s) have preds from fewer than "
              f"{n_ckpt} checkpoint(s); they will be averaged over what's "
              f"available.")

    # ── Write CSV ───────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header_keys = ",".join(key_cols)
    with open(output_path, "w") as f:
        # Always emit pred_std + n_seeds columns so downstream consumers
        # don't have to branch on n_ckpt. For n_ckpt == 1, pred_std is 0.
        f.write(f"{header_keys},prediction,pred_std,n_seeds\n")
        for key, vals in accum.items():
            arr = np.array(vals, dtype=np.float64)
            id_str = ",".join(str(k) for k in key)
            f.write(f"{id_str},{arr.mean():.6f},{arr.std(ddof=0):.6f},"
                    f"{len(arr)}\n")

    if args.save_per_seed and n_ckpt > 1:
        per_seed_dir = output_path.parent / (output_path.stem + "_per_seed")
        per_seed_dir.mkdir(exist_ok=True, parents=True)
        for col_idx, (_, _, tag) in enumerate(ckpts):
            with open(per_seed_dir / f"{tag}.csv", "w") as f:
                f.write(f"{header_keys},prediction\n")
                for key, vals in accum.items():
                    if col_idx < len(vals):
                        id_str = ",".join(str(k) for k in key)
                        f.write(f"{id_str},{vals[col_idx]:.6f}\n")
        print(f"[done] Per-seed CSVs in: {per_seed_dir}")

    print(f"\n[done] Saved {len(accum)} predictions from {n_ckpt} "
          f"checkpoint(s) to {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-c", "--config", required=True,
                   help="Inference YAML (model_params must match training)")
    p.add_argument("--output", required=True, help="Output CSV path")

    # Input source (pick one)
    p.add_argument("--manifest", default=None,
                   help="CSV manifest. Minimal form needs only compound_id, "
                        "kinase_A, kinase_B; structure paths are built from "
                        "--structures_root and a `label` column is optional. "
                        "Explicit path columns are honored when present.")
    p.add_argument("--test_dir", default=None,
                   help="Override dataset_params.precomputed_dirs.test "
                        "(only used when --manifest is not given)")

    # Structure-path resolution for minimal manifests
    p.add_argument("--structures_root", default=None,
                   help="Root dir holding uploaded structures. Paths are built "
                        "as <root>/<kinase>/<kinase>.pdb and "
                        "<root>/<kinase>/<compound_id>.sdf when the manifest "
                        "omits explicit path columns (default: 'inference_data').")
    p.add_argument("--protein_template", default=None,
                   help="Relative protein-pdb template; placeholders {kinase}, "
                        "{compound_id} (default: '{kinase}/{kinase}.pdb').")
    p.add_argument("--ligand_template", default=None,
                   help="Relative ligand-sdf template; placeholders {kinase}, "
                        "{compound_id} (default: '{kinase}/{compound_id}.sdf').")

    # Checkpoint selection (pick exactly one mode)
    p.add_argument("--ckpt", default=None,
                   help="Single explicit .ckpt path")
    p.add_argument("--seed_dir", default=None,
                   help="Single seed_<N>/ dir; auto-picks best ckpt inside it")
    p.add_argument("--seed_dirs", nargs="+", default=None,
                   help="Explicit list of seed_<N>/ dirs to ensemble")
    p.add_argument("--ensemble", action="store_true",
                   help="Auto-discover seed_*/ subdirs under save_dir "
                        "(or --parent_dir) and ensemble them")
    p.add_argument("--parent_dir", default=None,
                   help="With --ensemble: parent of seed_<N>/ dirs "
                        "(default: config['save_dir'])")

    # Modifiers
    p.add_argument("--pick_best", action="store_true",
                   help="When multiple seeds are found, keep only the one "
                        "with highest val_pearson instead of ensembling")
    p.add_argument("--save_per_seed", action="store_true",
                   help="With ensembling, also dump per-seed prediction CSVs")
    p.add_argument("--failed_output", default=None,
                   help="CSV collecting pairs skipped because a referenced "
                        "pdb/sdf file is missing (default: "
                        "<output>_failed_pairs.csv)")
    args = p.parse_args()
    main(args)