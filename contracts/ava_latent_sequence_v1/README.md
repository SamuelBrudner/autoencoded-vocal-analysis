# `ava_latent_sequence_v1`

This directory is the public, canonical producer contract for AVA latent
sequences. A conforming clip is a same-directory, same-stem `.npz` and `.json`
pair. `metadata.schema.json` governs the JSON sidecar; the exact numeric array
rules are listed in `docs/latent_sequence_export.md` and enforced by
`ava.models.latent_sequence_contract.validate_latent_sequence_pair`.

Every sidecar carries explicit recording metadata keys: `recording_id`,
`bird_id`, `dph`, `regime`, and `tutor_start_dph`. Unknown facts are encoded as
JSON `null`; neither producer nor consumer may reconstruct them from paths.
`audio_path` is a portable relative identity, never a workstation path.

`fixtures/` contains one valid pair and deliberately invalid cases shared by
producer and consumer tests. Consumers should pin the AVA commit and record the
SHA-256 of `metadata.schema.json`; they must not depend on a private control
repository.

Collection acceptance is a separate layer. The member manifest lists the clip
pairs, while an `ava_latent_sequence_export_acceptance_v1` record points to its
SHA-256 alongside dataset, configuration, checkpoint, and code SHA-256 values.
Keeping that record outside the member manifest avoids a circular self-hash.
