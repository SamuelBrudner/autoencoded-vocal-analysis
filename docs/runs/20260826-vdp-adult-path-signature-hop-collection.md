# Adult path-signature matched-hop latent collection

## Scope

This producer run supports the exploratory adult continuous path-signature
pilot. It uses the accepted `vdp-enc-002` checkpoint, the exact 216 clips in
registered export `vdp-export-002`, 30 ms acoustic windows, no ROI filtering,
and per-window energy. It changes only the inference hop. It does not compute
annotation associations or scientific outcomes.

## Frozen identities

- Dataset/panel (`vdp-ds-004`):
  `4e4bf15e9d8c4e99429c9e4a55fd29b2b0b1d61f6fb50580572425a4c9e3b59f`
- Exact member selection:
  `b7fc5cec5dd47d79e03911968fee61c7e1ee275c8b403459752b2b9ef290fccf`
- Encoder checkpoint (`vdp-enc-002`):
  `881ecbd0b0013c4ba9c69662790e45785050d17bb93d23d4443a008bf4ebee08`
- Frozen inference configuration:
  `4e0c1b4d34b6211567fd24cf7d51def53e0bb93e8f03d863a6f2b5b7ce08a48b`
- Producer implementation identity:
  `dc3be3b3ff2b3f12da447631ad857160927df305cb705a2223bb85f975c14dc4`
- Registered 10 ms anchor member manifest:
  `dc6eb1f0f8223a8d7582d6d690aa4e47d3ce82f7d3de052526857090f8ff3b01`

## Conformance results

| Hop | Clips | Windows | Protocol SHA-256 | Member manifest SHA-256 |
|---:|---:|---:|---|---|
| 30 ms | 216 | 75,520 | `02459389439c411df424f7b9bef9ca482e34327fdeacc5dbdbf49e487ceeb60b` | `42ac2c1017adb2d4203d244c4ab0924718de328fd59fe843e6a5eb1f599a61fb` |
| 15 ms | 216 | 150,925 | `36a4387b42ffd22a0b614044edcb9835c8b83c5750b4489bc968b38fea29aa54` | `c353e4bfde5eada195b5d3f4400b008c612b9ff9faddcfd2a0396a44a7e393c8` |
| 10 ms | 216 | 226,328 | `47393ffb168bea66a84b8b63f323df1e891320ecb499b7ae4269b20ce3281c15` | `dc6eb1f0f8223a8d7582d6d690aa4e47d3ce82f7d3de052526857090f8ff3b01` |
| 5.804988662 ms | 216 | 389,764 | `d306a92745a8568109e3a8dc08015aec93ef0e143fa2d7df8553b215c2578236` | `d9c47a6a29ceb7ca2a048b7c38aff4985e656f12027b066eb6de1fb1ec8d39f5` |

All alternate collections passed same-stem pairing, schema, dtype, finite-value,
timestamp monotonicity, dimensionality, energy, checkpoint, code, dataset, and
configuration checks. Membership is identical across hops by source audio
identity and clip ID. The collection manifest SHA-256 is
`488feafb452aa980b1cf9bdf1148ff01c0470790c212eee8a0fe997ace108747`.

## Runtime and durability

A target-free 12-clip benchmark at the densest hop took 32.57 seconds on local
CPU. Full local wall times were 161.50 seconds at 30 ms, 303.75 seconds at
15 ms, and 726.02 seconds at 5.804988662 ms. No cloud compute was launched.

The alternate-hop collection was uploaded with AES-256 server-side encryption
to the private versioned prefix
`s3://ava-birdsong-us-east-1-a1859d31/ava/birdsong/latent/vdp/adult-path-signature-hop-collection-20260826/`.
The prefix contains 1,312 objects and 148,617,678 bytes. A clean download of
the collection manifest reproduced its SHA-256 exactly. The existing registered
10 ms collection remains the immutable anchor rather than being duplicated.

This producer collection is eligible for draft registration as `vdp-export-005`.
Registration remains a human action, and no outcome analysis was performed in
this AVA run.
