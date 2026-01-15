# Birdsong Dataset Inventory

Path: `/Volumes/samsung_ssd/data/birdsong`

Context: Zebra finch song recordings from Ofer Tchernichovsi's lab.

## Summary
- Total size: 3.18 TiB (3,254.62 GiB)
- Total files: 5,174,439
- Audio: 5,169,286 `.wav` files (99.90% of all files)
- AppleDouble metadata: 4,659 `._*` files (macOS resource forks)
- Other file types: `<none>` (116), `.myd` (96), `.myi` (96), `.frm` (95), `.bmp` (82), `.db` (3), `.exe` (3), `.zip` (1), `.asv` (1), `.log` (1)

## Top-Level Directories
| Directory | Files | Size (GiB) |
| --- | ---: | ---: |
| Day 35 bells | 151,278 | 98.61 |
| day 35 Simple | 81,411 | 58.10 |
| day 43 samba | 790,644 | 762.12 |
| day 43 simple | 481,219 | 322.19 |
| day 60 bells | 147,727 | 98.98 |
| day 60 samba | 249,481 | 161.92 |
| day 60 simple | 36,887 | 24.63 |
| day 75 bells | 350,637 | 133.08 |
| day 75 samba | 269,382 | 131.98 |
| day 90 samba | 513,062 | 201.84 |
| day 90 simple | 209,665 | 151.86 |
| day43 Bells | 487,391 | 319.93 |
| day90 bells | 326,330 | 138.69 |
| isolates | 1,079,311 | 650.69 |

## Age Ranges (dph) by Directory
Derived from numeric folder names directly under bird ID directories (bird IDs look like `R658` or `R 203`).

| Directory | Bird IDs | Distinct dph | DPH range |
| --- | ---: | ---: | ---: |
| Day 35 bells | 2 | 97 | 34-140 |
| day 35 Simple | 1 | 0 | N/A |
| day 43 samba | 4 | 72 | 33-105 |
| day 43 simple | 3 | 73 | 39-112 |
| day 60 bells | 2 | 58 | 36-99 |
| day 60 samba | 3 | 58 | 36-99 |
| day 60 simple | 1 | 10 | 90-99 |
| day 75 bells | 4 | 98 | 34-131 |
| day 75 samba | 4 | 91 | 42-132 |
| day 90 samba | 5 | 99 | 35-133 |
| day 90 simple | 2 | 62 | 37-99 |
| day43 Bells | 7 | 69 | 31-99 |
| day90 bells | 4 | 63 | 37-99 |
| isolates | 13 | 87 | 35-666 |

## Notes
- The dataset is overwhelmingly `.wav` audio; other file types look like small database artifacts (e.g., `.myd/.myi/.frm`) and a handful of images/logs.
- There are AppleDouble `._*` files at the root and within folders; these are macOS metadata and can usually be ignored for analysis.
- Folder naming (per your guidance): `bells`, `simple`, `samba`, and `isolates` refer to different artificial tutoring regimes (song models). `Day **` indicates the day tutoring begins. Numeric subfolders inside these directories are the dph (days post-hatch) when the audio was recorded.
- `day 35 Simple` uses date-stamped session folders under `R205` (e.g., `205Oct_31_15_33`) rather than standalone numeric dph folders, so age ranges are not inferable from folder names alone.
- `day 35 Simple` contains 16 session directories (from `205Oct_26_10_40` through `205Jan_01_14_44`) and 81,390 `.wav` files for bird `R205`.
- `day 35 Simple` filenames encode recording timestamps as `bird205_<index>_on_<Mon>_<DD>_<HH>_<MM>.wav` (recordings span Oct 26 through Jan 10); dph is not encoded in names.
- `isolates` includes a small set of high numeric folders (>200: 276, 284, 294, 332, 333, 334, 344, 346, 427, 666); confirm whether these represent dph or a different labeling scheme.
