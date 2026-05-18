# Developmental Replication Input Inventory

## Summary

- Birds requested: 11.
- Birds with audio dirs: 11 / 11.
- Birds with ROI artifacts: 11 / 11.
- Birds with latent sequences: 1 / 11.
- Ready for local full rebuild: 1 / 11.

## Bird Status

- `PK249`: ready_for_full_rebuild; manifest files=95402, audio wavs=95402, ROI dirs=58/58, latent npz=190168.
- `R426`: missing_latents; manifest files=160735, audio wavs=160735, ROI dirs=51/51, latent npz=0.
- `R467`: missing_latents; manifest files=158905, audio wavs=158905, ROI dirs=53/53, latent npz=0.
- `R404`: missing_latents; manifest files=146068, audio wavs=146068, ROI dirs=50/50, latent npz=0.
- `R150`: missing_latents; manifest files=130668, audio wavs=130668, ROI dirs=95/95, latent npz=0.
- `R493`: missing_latents; manifest files=127705, audio wavs=127705, ROI dirs=56/56, latent npz=0.
- `R470`: missing_latents; manifest files=73908, audio wavs=73908, ROI dirs=54/54, latent npz=0.
- `R203`: missing_latents; manifest files=48509, audio wavs=48510, ROI dirs=39/39, latent npz=0.
- `R425`: missing_latents; manifest files=77520, audio wavs=77520, ROI dirs=53/53, latent npz=0.
- `R229`: missing_latents; manifest files=66293, audio wavs=66293, ROI dirs=53/53, latent npz=0.
- `R122`: missing_latents; manifest files=52654, audio wavs=52654, ROI dirs=40/40, latent npz=0.

## Next Export Blockers

Latent sequences are missing for `R122`, `R150`, `R203`, `R229`, `R404`, `R425`, `R426`, `R467`, `R470`, `R493`. Stage ROI artifacts and export latents with the same checkpoint/config used for PK249 before rerunning `scripts/analyze_developmental_replication.py` in full-rebuild mode.

## Artifacts

- `input_inventory`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260518-145000-post-roi-recovery-inventory/developmental_input_inventory.json`
- `cohort_manifest`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260518-145000-post-roi-recovery-inventory/developmental_cohort_manifest.json`
