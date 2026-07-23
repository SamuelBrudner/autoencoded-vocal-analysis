# `ava_latent_sequence_v1`

This directory is the public, canonical producer contract for AVA latent
sequences. A conforming clip is a same-directory, same-stem `.npz` and `.json`
pair. `metadata.schema.json` governs the JSON sidecar; the exact numeric array
rules are listed in `docs/latent_sequence_export.md` and enforced by
`ava.models.latent_sequence_contract.validate_latent_sequence_pair`.

`fixtures/` contains one valid pair and deliberately invalid cases shared by
producer and consumer tests. Consumers should pin the AVA commit and record the
SHA-256 of `metadata.schema.json`; they must not depend on a private control
repository.

Collection acceptance is a separate layer. The member manifest lists the clip
pairs, while an `ava_latent_sequence_export_acceptance_v1` record points to its
SHA-256 alongside dataset, configuration, checkpoint, and code SHA-256 values.
Keeping that record outside the member manifest avoids a circular self-hash.
