"""
AVA: Autoencoded Vocal Analysis

Contents
--------

::

	ava
	│
	├── data
	│	└── data_container
	├── models
	│	├── vae_dataset
	│	├── vae
	│	├── shotgun_vae_dataset
	│	└── window_vae_dataset
	├── plotting
	│	├── grid_plot
	│	├── latent_projection
	│	├── mmd_plots
	│	├── shotgun_movie
	│	└── tooltip_plot
	├── preprocessing
	│	├── preprocess
	│	└── utils
	└── segmenting
		├── amplitude_segmenting
		├── refine_segments
		├── segment
		├── template_segmentation
		└── utils
"""
import os
import tempfile


def _configure_numba_cache_dir():
	"""
	Provide a writable numba cache directory when none is configured.

	This avoids runtime failures when importing UMAP/pynndescent in
	readonly-style conda environments.
	"""
	if os.environ.get("NUMBA_CACHE_DIR"):
		return
	cache_dir = os.path.join(tempfile.gettempdir(), "ava_numba_cache")
	try:
		os.makedirs(cache_dir, exist_ok=True)
		if os.access(cache_dir, os.W_OK):
			os.environ["NUMBA_CACHE_DIR"] = cache_dir
	except OSError:
		# Keep default behavior if the fallback path cannot be created.
		pass


_configure_numba_cache_dir()

__version__ = "0.3.1"
