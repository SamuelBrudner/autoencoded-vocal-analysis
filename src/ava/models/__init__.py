"""
AVA package for training a VAE

Contains
--------
`ava.models.vae`
	Defines the variational autoencoder (VAE).
`ava.models.lightning_vae`
	Lightning training utilities for the VAE.
`ava.models.vae_dataset`
	Feeds syllable data to the VAE.
`ava.models.shotgun_vae_dataset`
	Feeds random data to the shotgun VAE.
`ava.models.fixed_window_config`
	Structured configuration for fixed-window VAE experiments.
`ava.models.window_vae_dataset`
	Legacy name for shotgun VAE datasets.
`ava.models.utils`
	Useful functions related to the `ava.models` subpackage.
"""

__all__ = [
	"lightning_vae",
	"fixed_window_config",
	"optuna_sweep",
	"shotgun_vae_dataset",
	"utils",
	"vae",
	"vae_dataset",
	"window_vae_dataset",
]
