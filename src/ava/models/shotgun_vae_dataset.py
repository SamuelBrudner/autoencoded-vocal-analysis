"""
Shotgun VAE dataset utilities.

This module is the preferred public import for the windowed (shotgun) VAE
datasets. It aliases the legacy ``window_vae_dataset`` implementation to keep
backwards compatibility while presenting the updated naming.
"""
from __future__ import annotations

from ava.models.window_vae_dataset import (  # noqa: F401
	DEFAULT_WARP_PARAMS,
	FixedWindowDataset,
	WarpedWindowDataset,
	get_fixed_window_data_loaders,
	get_warped_window_data_loaders,
	get_window_partition,
)
from ava.models.manifest_window_dataset import (  # noqa: F401
	ManifestFixedWindowDataset,
	get_manifest_fixed_window_data_loaders,
)


FixedShotgunDataset = FixedWindowDataset
WarpedShotgunDataset = WarpedWindowDataset
get_shotgun_partition = get_window_partition
get_fixed_shotgun_data_loaders = get_fixed_window_data_loaders
get_warped_shotgun_data_loaders = get_warped_window_data_loaders

__all__ = [
	"DEFAULT_WARP_PARAMS",
	"FixedShotgunDataset",
	"FixedWindowDataset",
	"ManifestFixedWindowDataset",
	"WarpedShotgunDataset",
	"WarpedWindowDataset",
	"get_manifest_fixed_window_data_loaders",
	"get_fixed_shotgun_data_loaders",
	"get_fixed_window_data_loaders",
	"get_shotgun_partition",
	"get_window_partition",
	"get_warped_shotgun_data_loaders",
	"get_warped_window_data_loaders",
]
