"""
Latent-space evaluation utilities.

These helpers compute invariance metrics between paired views and
self-retrieval accuracy in latent space.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

from ava.models.vae import DEFAULT_INPUT_SHAPE, VAE


def _extract_pair(batch) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Return (base, aug) tensors from a batch of paired views."""
	if isinstance(batch, dict):
		base = batch.get("x")
		aug = batch.get("x_aug")
		if base is None or aug is None:
			raise ValueError("Dict batch must contain 'x' and 'x_aug' keys.")
		return base, aug
	if isinstance(batch, (tuple, list)) and len(batch) == 2:
		return batch[0], batch[1]
	raise ValueError("Expected a paired batch in tuple/list or dict form.")


def load_vae_from_checkpoint(
		checkpoint_path: str,
		device: str = "auto",
) -> VAE:
	"""
	Load a VAE checkpoint and return the initialized model.

	Parameters
	----------
	checkpoint_path : str
		Path to the ``.tar`` checkpoint saved by ``VAE.save_state``.
	device : {"cpu", "cuda", "auto"}, optional
		Device to place the model on.
	"""
	checkpoint = torch.load(checkpoint_path, map_location="cpu")
	z_dim = int(checkpoint.get("z_dim", 32))
	input_shape = tuple(checkpoint.get("input_shape", DEFAULT_INPUT_SHAPE))
	posterior_type = checkpoint.get("posterior_type", "lowrank")
	conv_arch = checkpoint.get("conv_arch", "plain")
	decoder_type = checkpoint.get("decoder_type", "convtranspose")
	learn_observation_scale = checkpoint.get("learn_observation_scale", False)
	model_precision = checkpoint.get("model_precision", 10.0)
	model = VAE(
		z_dim=z_dim,
		input_shape=input_shape,
		posterior_type=posterior_type,
		conv_arch=conv_arch,
		decoder_type=decoder_type,
		learn_observation_scale=learn_observation_scale,
		model_precision=model_precision,
		device_name=device,
		build_optimizer=False,
	)
	model.load_state(checkpoint_path)
	model.eval()
	return model


def collect_latent_pairs(
		model: VAE,
		loader: Iterable,
		max_samples: Optional[int] = None,
		device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Collect paired latent means from a loader that yields paired views.

	Returns
	-------
	base_latent : numpy.ndarray
		Latent means for the base view.
	aug_latent : numpy.ndarray
		Latent means for the augmented view.
	"""
	if device is None:
		device = getattr(model, "device", torch.device("cpu"))
	base_latent = []
	aug_latent = []
	seen = 0
	model.eval()
	with torch.inference_mode():
		for batch in loader:
			base, aug = _extract_pair(batch)
			base = base.to(device)
			aug = aug.to(device)
			mu_base, _, _ = model.encode(base)
			mu_aug, _, _ = model.encode(aug)
			mu_base = mu_base.detach().cpu()
			mu_aug = mu_aug.detach().cpu()
			try:
				mu_base = mu_base.numpy()
				mu_aug = mu_aug.numpy()
			except RuntimeError as exc:
				if "Numpy is not available" not in str(exc):
					raise
				mu_base = np.asarray(mu_base.tolist(), dtype=np.float32)
				mu_aug = np.asarray(mu_aug.tolist(), dtype=np.float32)
			if max_samples is not None and seen + len(mu_base) > max_samples:
				keep = max_samples - seen
				if keep <= 0:
					break
				mu_base = mu_base[:keep]
				mu_aug = mu_aug[:keep]
			base_latent.append(mu_base)
			aug_latent.append(mu_aug)
			seen += len(mu_base)
			if max_samples is not None and seen >= max_samples:
				break
	if not base_latent:
		raise ValueError("No latent pairs were collected from the loader.")
	return np.concatenate(base_latent, axis=0), np.concatenate(aug_latent, axis=0)


def compute_latent_invariance(
		base_latent: np.ndarray,
		aug_latent: np.ndarray,
) -> dict:
	"""
	Compute summary statistics for latent invariance (pairwise distances).
	"""
	if base_latent.shape != aug_latent.shape:
		raise ValueError("Latent arrays must have the same shape.")
	if base_latent.size == 0:
		raise ValueError("Latent arrays must be non-empty.")
	deltas = base_latent - aug_latent
	distances = np.linalg.norm(deltas, axis=1)
	return {
		"mean": float(np.mean(distances)),
		"median": float(np.median(distances)),
		"std": float(np.std(distances)),
		"p05": float(np.percentile(distances, 5)),
		"p95": float(np.percentile(distances, 95)),
	}


def compute_self_retrieval(
		base_latent: np.ndarray,
		aug_latent: np.ndarray,
		chunk_size: int = 1024,
) -> dict:
	"""
	Compute self-retrieval accuracy for paired latent representations.
	"""
	if base_latent.shape != aug_latent.shape:
		raise ValueError("Latent arrays must have the same shape.")
	if base_latent.size == 0:
		raise ValueError("Latent arrays must be non-empty.")
	if chunk_size <= 0:
		raise ValueError("chunk_size must be positive.")
	base_latent = np.asarray(base_latent, dtype=np.float32)
	aug_latent = np.asarray(aug_latent, dtype=np.float32)
	n_samples = base_latent.shape[0]
	aug_norms = np.sum(aug_latent ** 2, axis=1)
	top1_hits = 0
	top5_hits = 0
	ranks = []
	for start in range(0, n_samples, chunk_size):
		stop = min(start + chunk_size, n_samples)
		chunk = base_latent[start:stop]
		chunk_norms = np.sum(chunk ** 2, axis=1, keepdims=True)
		distances = chunk_norms + aug_norms[None, :] - 2.0 * chunk.dot(aug_latent.T)
		distances = np.maximum(distances, 0.0)
		nearest = np.argmin(distances, axis=1)
		indices = np.arange(start, stop)
		top1_hits += int(np.sum(nearest == indices))
		correct_dist = distances[np.arange(stop - start), indices]
		rank = 1 + np.sum(distances < correct_dist[:, None], axis=1)
		ranks.append(rank)
		top5_hits += int(np.sum(rank <= 5))
	ranks = np.concatenate(ranks, axis=0)
	return {
		"top1": float(top1_hits / n_samples),
		"top5": float(top5_hits / n_samples),
		"mean_rank": float(np.mean(ranks)),
		"median_rank": float(np.median(ranks)),
	}


def evaluate_latent_metrics(
		model: VAE,
		loaders: Sequence[Iterable],
		max_samples: Optional[int] = None,
		chunk_size: int = 1024,
		device: Optional[torch.device] = None,
) -> dict:
	"""
	Collect latent pairs from one or more loaders and compute metrics.
	"""
	base_chunks = []
	aug_chunks = []
	remaining = max_samples
	for loader in loaders:
		if loader is None:
			continue
		base_latent, aug_latent = collect_latent_pairs(
			model,
			loader,
			max_samples=remaining,
			device=device,
		)
		base_chunks.append(base_latent)
		aug_chunks.append(aug_latent)
		if remaining is not None:
			remaining -= base_latent.shape[0]
			if remaining <= 0:
				break
	if not base_chunks:
		raise ValueError("No loaders produced latent pairs for evaluation.")
	base_latent = np.concatenate(base_chunks, axis=0)
	aug_latent = np.concatenate(aug_chunks, axis=0)
	invariance = compute_latent_invariance(base_latent, aug_latent)
	retrieval = compute_self_retrieval(
		base_latent,
		aug_latent,
		chunk_size=chunk_size,
	)
	results = {
		"num_pairs": int(base_latent.shape[0]),
		"latent_invariance": invariance,
		"self_retrieval": retrieval,
		"latent_invariance_mean": invariance["mean"],
		"self_retrieval_top1": retrieval["top1"],
	}
	return results
