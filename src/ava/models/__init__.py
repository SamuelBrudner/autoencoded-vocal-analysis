"""
AVA package for training a VAE

Contains
--------
`ava.models.vae`
	Defines the variational autoencoder (VAE).
`ava.models.lightning_vae`
	Lightning training utilities for the VAE.
`ava.models.lightning_sequence_vae`
	Lightning training utilities for recurrent sequence VAEs.
`ava.models.vae_dataset`
	Feeds syllable data to the VAE.
`ava.models.shotgun_vae_dataset`
	Feeds random data to the shotgun VAE.
`ava.models.sequence_window_dataset`
	Feeds ordered whole-file window sequences to recurrent models.
`ava.models.fixed_window_config`
	Structured configuration for fixed-window VAE experiments.
`ava.models.window_vae_dataset`
	Legacy name for shotgun VAE datasets.
`ava.models.vrnn`
	Variational recurrent latent model for contiguous sequences.
`ava.models.blocks`
	Residual block primitives for spectrogram models.
`ava.models.latent_metrics`
	Latent invariance and self-retrieval metrics.
`ava.models.utils`
	Useful functions related to the `ava.models` subpackage.
"""

__all__ = [
	"blocks",
	"latent_metrics",
	"lightning_sequence_vae",
	"lightning_vae",
	"fixed_window_config",
	"optuna_sweep",
	"sequence_window_dataset",
	"shotgun_vae_dataset",
	"utils",
	"vae",
	"vae_dataset",
	"vrnn",
	"window_vae_dataset",
]
