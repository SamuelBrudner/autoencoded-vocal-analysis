# Birdsong Manifest Format

AVA uses a **manifest** to drive birdsong ROI generation, training, and export
scripts in a reproducible, stratified way.

The canonical manifest is a **JSON file** produced by:

```bash
python scripts/build_birdsong_manifest.py \
  --metadata data/metadata/birdsong/birdsong_files.parquet \
  --roi-root /path/to/roi_root \
  --out data/manifests/birdsong_manifest.json
```

Dependency note:
- Manifest generation requires a parquet reader in the active environment.
  The project conda env specs include `pyarrow`; verify it is installed before running manifest workflows inside `ava`.

## Top-Level Schema

Required keys:
- `created_utc`: ISO-8601 UTC timestamp string
- `metadata_path`: path to the parquet metadata used to build the manifest
- `root`: dataset root used when resolving absolute `audio_dir`
- `roi_root`: ROI root used when resolving absolute `roi_dir`
- `seed`: integer RNG seed
- `train_fraction`: float in (0, 1)
- `train`: list of entries (see below)
- `test`: list of entries (see below)

Recommended keys (produced by the builder):
- `filters`: record of builder filters/limits
- `bird_sets`: per-regime list of bird IDs in train/test
- `summary`: counts for train/test (birds, directories, files)

## Entry Schema (Per Audio Directory)

Each entry in `train` or `test` must include:
- `audio_dir_rel`: path relative to `root` (or `"."`)
- `audio_dir`: absolute audio directory path
- `roi_dir`: absolute ROI directory path
- `bird_id_norm`: normalized bird ID (whitespace removed, uppercased)
- `bird_id_raw`: raw bird ID directory name
- `regime`: regime label (e.g., `bells`, `simple`, `samba`, `isolates`)
- `dph`: integer DPH when present, otherwise null
- `session_label`: non-numeric session label path component(s), otherwise null
- `num_files`: integer number of wav files found under `audio_dir`
- `split`: `"train"` or `"test"`

Additional informational fields currently included by the builder:
- `top_dir`
- `pre_bird_path`

## Consumption Notes

- Scripts that consume manifests generally accept `--audio-root` and `--roi-root`
  overrides. If an entry is missing `audio_dir` or `roi_dir`, those overrides
  are used to resolve absolute paths from `audio_dir_rel`.
- Manifests are directory-level: downstream code enumerates `.wav` files inside
  each `audio_dir` and expects ROI text files in `roi_dir` with matching
  basenames (e.g., `foo.wav` → `foo.txt`).
