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

## Notes
- The dataset is overwhelmingly `.wav` audio; other file types look like small database artifacts (e.g., `.myd/.myi/.frm`) and a handful of images/logs.
- There are AppleDouble `._*` files at the root and within folders; these are macOS metadata and can usually be ignored for analysis.
- Folder naming (per your guidance): `bells`, `simple`, `samba`, and `isolates` refer to different artificial tutoring regimes (song models). `Day **` indicates the day tutoring begins. Numeric subfolders inside these directories are the dph (days post-hatch) when the audio was recorded.
