# Birdsong ROI Layout Contract

AVA birdsong workflows assume a **mirrored directory layout** between audio and
ROI outputs.

## Directory Mapping

For each leaf `audio_dir` that contains `.wav` files, there is a corresponding
`roi_dir` containing ROI `.txt` files.

The recommended convention is:

```
<AUDIO_ROOT>/<audio_dir_rel>/*.wav
<ROI_ROOT>/<audio_dir_rel>/*.txt
```

where `audio_dir_rel` is a path relative to the dataset root stored in the
birdsong metadata/manifest.

## File Naming

For each wav file:

```
<audio_dir>/foo.wav
```

the ROI file must be:

```
<roi_dir>/foo.txt
```

## ROI File Contents

ROI files are plain text with one segment per line:

```
<onset_sec> <offset_sec>
```

Lines beginning with `#` are ignored.

## Validation

Use `scripts/validate_birdsong_rois.py` to check for missing or empty ROI files
before running training or export.

