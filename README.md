# 3D-PLKS 

3D-PLKS predicts the **relative binding-affinity difference** between two kinases
for the same compound, directly from their 3D docked protein–ligand complexes.
For a pair `(compound, kinase_A, kinase_B)` it outputs a single number, **ΔpAct =
pAct(kinase_A) − pAct(kinase_B)**

To get a prediction you need provide two things:

1. the docked structure files for the kinases and compounds you care about,
2. a small CSV listing which pairs to score.

---

## 1. Environment

Activate the project conda environment (it already contains the dependencies):

```bash
conda env create -f environment.yml
conda activate 3DPLKS
```

---

## 2. Prepare your inputs

### 2a. Structure files

The model reads, for each pair, one protein pocket `.pdb` per kinase and one
docked ligand `.sdf` per (kinase, compound). Organize them under a single root
directory using this layout:

```
inference_data/task1/
├── P23458/                 # kinase = UniProt accession
│   ├── P23458.pdb          # the kinase pocket structure
│   ├── 99960618.sdf        # compound 99960618 docked into P23458
│   └── 99945275.sdf
├── P52333/
│   ├── P52333.pdb
│   ├── 99960618.sdf        # SAME compound, docked into P52333 (different pose)
│   └── ...
└── ...
```

Two rules that matter:

- **Folder name = the kinase ID** used in the CSV (UniProt accessions for the JAK
  panel are listed in the Appendix).
- **The ligand `.sdf` is the docked pose, and it is kinase-specific.** The same
  compound must have its own `.sdf` inside *each* kinase folder it is compared in,
  because the pose differs per pocket. These poses must be generated beforehand
  (e.g. by docking); the inference tool consumes them, it does not dock for you.

### 2b. The pairs manifest (CSV)

This is all a user needs to write. **Three columns, nothing else:**

```csv
compound_id,kinase_A,kinase_B
99960618,P23458,P52333
99945275,P29597,P52333
```

| Column | Meaning |
| --- | --- |
| `compound_id` | Your identifier for the compound (matches the `.sdf` filename stem). |
| `kinase_A` | Kinase ID of the first target (matches a structure folder). |
| `kinase_B` | Kinase ID of the second target. |

The four file paths are built automatically from these three values plus
`--structures_root`, now is `inference_data/'task_id/'`

Optional, for advanced use:

- `label` — include it only if you have the true ΔpAct and want it carried along
  for your own evaluation; it does not affect predictions.
- `proteinA_pdb`, `ligandA_sdf`, `proteinB_pdb`, `ligandB_sdf` — explicit paths.
  If present and non-blank they override the auto-built paths (so older full
  manifests still work). Blank cells fall back to the convention.

---

## 3. Run prediction

```bash
python inference_live.py \
  -c config_files/config_inference_live.yaml \
  --manifest inference_data/task1/case_study_pairs.csv \
  --structures_root inference_data/task1 \
  --ensemble \
  --output predictions_case.csv
```

Run it from the directory where your relative paths are valid (paths in the CSV
and `--structures_root` are resolved against the current working directory).

| Argument | Purpose |
| --- | --- |
| `-c, --config` | Inference YAML (model settings must match training). Required. |
| `--manifest` | Your pairs CSV. |
| `--structures_root` | Folder holding the structures (default `inference_data`). |
| `--output` | Output CSV path. Required. |
| `--ensemble` | Average all `seed_*/` checkpoints found under the config's `save_dir` (or `--parent_dir`). Recommended. |
| `--ckpt` | Use one explicit `.ckpt` instead of an ensemble. |
| `--seed_dir` / `--seed_dirs` | Use one / several specific seed folders. |
| `--pick_best` | From several seeds, keep only the highest `val_pearson` one. |
| `--save_per_seed` | Also write each seed's individual predictions. |
| `--protein_template` / `--ligand_template` | Override the file-naming convention. |
| `--failed_output` | Where to write skipped pairs (default `<output>_failed_pairs.csv`). |

---

## 4. Read the output

`predictions_case.csv`:

```csv
compound_id,kinase_A,kinase_B,prediction,pred_std,n_seeds
99960618,P23458,P52333,2.104213,0.118402,3
```

| Column | Meaning |
| --- | --- |
| `prediction` | Predicted **ΔpAct = pAct(kinase_A) − pAct(kinase_B)**. |
| `pred_std` | Standard deviation across ensemble seeds (0 for a single checkpoint). A rough confidence indicator. |
| `n_seeds` | How many checkpoints were averaged. |


## 5. Troubleshooting

**`error: unrecognized arguments: python inference_live.py`**
You pasted the command twice onto one line. Run a single command. With line
continuations, make sure each `\` is the last character on its line (a stray `\ `
glued to the next word causes this too).

**Lots of pairs land in `_failed_pairs.csv`**
The structures aren't where the tool expects. Check that `--structures_root`
points at the folder that actually contains the per-kinase subfolders. For
example, if your files live in `inference_data/task1/P23458/...`, pass
`--structures_root inference_data/task1`, not `--structures_root inference_data`.
Then confirm folder names equal the kinase IDs in the CSV and that each compound's
`.sdf` exists inside *every* kinase folder it is paired against.

**`Manifest is missing required column(s)`**
The CSV header must contain `compound_id`, `kinase_A`, `kinase_B` (exact names).

**Predictions look offset or noisy on a custom set**
The model expects pockets and docked poses prepared the same way as in training.
Use the same pocket-extraction and docking protocol; mismatched preparation
degrades accuracy.

---