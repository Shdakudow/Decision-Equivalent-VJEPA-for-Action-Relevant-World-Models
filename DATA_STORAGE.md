# Data and Artifact Storage

The GitHub repository is intentionally source-first. It includes code,
configuration files, compact summary/statistical CSVs, LaTeX tables, plots, and
the paper source. It does not include datasets, model checkpoints, experiment
output directories, or large per-sample CSVs.

## Excluded Artifacts

| Artifact class | Local path | Recommended storage |
|---|---|---|
| Raw and encoded datasets | `data/` | Zenodo, OSF, institutional storage, or a private object store |
| Predictor and policy runs | `outputs/` | External artifact archive; publish selected final checkpoints only |
| Progress-head checkpoints | `checkpoints/` | External artifact archive |
| Large per-sample metrics | `results/*.csv` | Compressed external archive |

GitHub rejects ordinary Git objects above 100 MB. Several raw result files in
this project are 160-439 MB, so they must not be committed directly. Git LFS is
possible for selected checkpoints, but a versioned research archive such as
Zenodo is preferable for a paper release.

## Included Evidence

The portable repository retains:

- `results/*_summary.csv`
- `results/*_statistics.csv`
- `results/*_paired.csv`
- `results/*_manifest.csv`
- `tables/*.tex`
- `plots/*.png`

These files are sufficient to inspect every number reported in `main.tex`.
They are not sufficient to retrain models without the external datasets.

## Restoring External Artifacts

After downloading an external artifact bundle, restore its top-level
directories into the repository:

```text
de-vjepa/
  data/
  outputs/
  checkpoints/
  results/
```

Do not replace compact summary files with older versions. Raw files and
checkpoints should be copied alongside the tracked summaries.

## Required Dataset Fields

MetaWorld encoded archives must contain aligned arrays for observations,
actions, task IDs, episode IDs, success flags, and latents. Shifted archives
must preserve the exact sample order of the clean archive. PushT archives use
the same alignment convention.

The demonstrations currently do not contain reward-to-go or task distance.
Consequently, the planning head is trained on normalized trajectory position
and is described in the paper as a progress head, not a true value function.

## Publication Checklist

Before publishing an artifact archive:

1. Remove credentials, machine-specific paths, and unrelated files.
2. Include dataset provenance and redistribution terms.
3. Include SHA-256 checksums for each archive.
4. Record the Git commit corresponding to the artifacts.
5. Assign a stable DOI or release URL and add it to the README and paper.

No software or dataset license has been selected in this package. Choose one
only after confirming that all upstream MetaWorld, PushT, and pretrained-model
terms permit the intended redistribution.
