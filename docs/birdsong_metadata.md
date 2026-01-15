# Birdsong Metadata Database

This metadata database indexes every file under `/Volumes/samsung_ssd/data/birdsong` so we can slice the dataset flexibly (by regime, tutor start day, bird ID, dph, session, or recording time) without moving raw audio.

## Outputs
- `data/metadata/birdsong/birdsong_files.parquet` (file-level metadata)
- `data/metadata/birdsong/birdsong_files_summary.json` (row counts and basic stats)

These artifacts are local (ignored by git) and can be regenerated at any time.

## Build
```bash
python scripts/build_birdsong_metadata.py
```

Optional flags:
- `--root` to point at a different dataset location
- `--out` to choose a different Parquet filename
- `--summary` for the summary JSON path

## Key Columns
- `rel_path`: path to the file relative to the dataset root
- `audio_dir_rel`: parent directory (relative) containing the file
- `top_dir`: top-level directory (e.g., `day 43 samba`, `day90 bells`)
- `regime`: `bells`, `simple`, `samba`, or `isolates`
- `tutor_start_day`: parsed from the top-level folder name
- `bird_id_raw`: bird ID directory name (e.g., `R658`, `R 203`, `pk244`)
- `bird_id_norm`: normalized bird ID (whitespace removed, uppercased)
- `dph`: numeric directory under a bird ID, when present
- `session_label`: non-numeric directory name(s) between bird ID and dph (e.g., `205Oct_31_15_33`, `Later`)
- `recording_month`, `recording_day`, `recording_hour`, `recording_min`, `recording_sec`: parsed from filenames containing `_on_`
- `file_index`: numeric token immediately before `_on_` in the filename
- `file_ext`, `size_bytes`, `is_audio`, `is_appledouble`

Notes:
- `dph` is inferred from numeric folders directly under bird IDs. If a directory uses date-stamped sessions (e.g., `day 35 Simple`), dph is `NULL` and timestamps come from folder/file names.
- Use `rel_path` to map files into object storage (e.g., `s3://bucket/birdsong/` + `rel_path`) for cloud training.

## Example Queries
Pandas:
```python
import pandas as pd

df = pd.read_parquet('data/metadata/birdsong/birdsong_files.parquet')
# Example: day 43 samba, dph 50
subset = df[(df.regime == 'samba') & (df.tutor_start_day == 43) & (df.dph == 50)]
```

DuckDB (if installed):
```sql
SELECT COUNT(*)
FROM 'data/metadata/birdsong/birdsong_files.parquet'
WHERE regime = 'bells' AND tutor_start_day = 35 AND dph BETWEEN 60 AND 90;
```
