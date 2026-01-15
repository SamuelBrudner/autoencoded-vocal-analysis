Architecture Review and Modernization Roadmap
=============================================

This document captures the current AVA architecture and a conservative
modernization roadmap. It is intended to guide incremental improvements without
changing the scientific behavior.

Current Architecture
--------------------

Package layout
^^^^^^^^^^^^^^

* :code:`ava.segmenting` provides syllable detection, segmentation utilities,
  and optional refinement workflows.
* :code:`ava.preprocessing` builds spectrograms (log-mel or linear) and writes
  fixed-size arrays for training and analysis.
* :code:`ava.models` contains the VAE implementation and PyTorch datasets for
  syllable and shotgun training.
* :code:`ava.data` provides the :code:`DataContainer` for linking audio,
  segments, spectrograms, features, and projection outputs.
* :code:`ava.plotting` wraps matplotlib and bokeh visualizations for latent
  exploration and analysis.
* :code:`src/timescale_analysis.py` is a standalone analysis utility that
  computes mel-band autocorrelation timescales.

Data flow
^^^^^^^^^

1. Raw audio (.wav) lives in per-animal folders.
2. Segmenting writes onset and offset files (.txt) using amplitude or template
   algorithms from :code:`ava.segmenting`.
3. Preprocessing converts segments into spectrograms and stores them in HDF5
   files along with metadata.
4. Modeling trains :code:`ava.models.vae.VAE` using HDF5-backed datasets or
   windowed, on-the-fly spectrograms for shotgun training.
5. :code:`DataContainer` aggregates spectrograms, projections, and external
   features to power the analysis and plotting utilities.

Key artifacts and formats
^^^^^^^^^^^^^^^^^^^^^^^^^

* Audio input: .wav
* Segments: .txt files with onset/offset pairs
* Spectrograms and projections: .hdf5 files
* Features from external tools: .csv or .txt tables
* Model checkpoints: .tar files written by :code:`ava.models.vae.VAE`

Dependencies and integrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* PyTorch for modeling and training
* NumPy, SciPy, and matplotlib for preprocessing and analysis
* h5py for spectrogram and projection storage
* umap-learn and scikit-learn for projections
* affinewarp for time warping in shotgun workflows
* bokeh for interactive plotting

Constraints and risks
^^^^^^^^^^^^^^^^^^^^^

* Parameter dictionaries are shared across modules without a central schema,
  which makes validation and reuse harder.
* HDF5 is the primary storage format, so schema drift or file-level
  inconsistencies can be painful to debug.
* The VAE assumes 128x128 spectrograms; the architecture is hard-coded in
  :code:`ava.models.vae`.
* Some workflows are interactive (parameter tuning functions), which makes
  them harder to automate or reproduce.
* Test coverage is minimal and does not cover the full training pipeline.
* The :code:`affinewarp` dependency is pulled from git, which can make
  installation brittle.

Modernization Roadmap
---------------------

Phase 0: Documentation and guardrails (low risk)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Document parameter schemas for segmenting, preprocessing, and datasets in
  one place.
* Add small tests for segmentation IO, spectrogram generation, and HDF5
  read/write behavior.
* Add type hints for public APIs and clearer error messages on bad params.

Phase 1: Dependency and IO hygiene (moderate)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Make :code:`affinewarp` optional and prefer
  :code:`ava.preprocessing.warping` when possible.
* Move dependency metadata into :code:`pyproject.toml` (PEP 621) or align
  :code:`setup.py` with modern packaging tooling.
* Centralize file naming and HDF5 schema helpers to avoid divergence between
  modules.
* Add a lightweight CLI (segment, preprocess, train) that wraps existing
  functions without changing behavior.

Phase 2: Architecture upgrades (higher impact)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Replace HDF5 with chunked formats (for example, Zarr) or provide a pluggable
  storage layer.
* Refactor datasets to stream from disk and support variable spectrogram
  sizes.
* Replace interactive parameter tuning with config files and reproducible
  pipelines.
* Expand modeling to support newer architectures while keeping the current VAE
  as the default.

Near-term backlog candidates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Rename "window VAE" to "shotgun VAE" in code and docs for clarity.
* Remove the affinewarp dependency by porting warping to
  :code:`ava.preprocessing.warping`.
* Improve :code:`DataContainer` subset selection and feature import (see
  :code:`to_do.md`).
