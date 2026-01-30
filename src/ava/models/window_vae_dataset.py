"""
Methods for feeding randomly sampled spectrogram data to the shotgun VAE.

Note
----
Prefer importing from ``ava.models.shotgun_vae_dataset``; this module is
retained for backwards compatibility.

Meant to be used with `ava.models.vae.VAE`.

TO DO
-----
- replace `affinewarp` with `ava.preprocessing.warping`

"""
__date__ = "August 2019 - November 2020"


try:
	from affinewarp import PiecewiseWarping
except ModuleNotFoundError:
	PiecewiseWarping = None
from collections import OrderedDict
import hashlib
import json
import tempfile
import h5py
import multiprocessing
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

from ava.models.utils import numpy_to_tensor, _get_wavs_from_dir, \
		_get_specs_and_amplitude_traces


DEFAULT_WARP_PARAMS = {
	'n_knots': 0, # number of pieces minus one in the piecwise linear warp
	'warp_reg_scale': 1e-2, # penalizes distance of warp to identity line
	'smoothness_reg_scale': 1e-1, # penalizes L2 norm of warp second derivatives
	'l2_reg_scale': 1e-7, # penalizes L2 norm of warping template
}
"""Default time-warping parameters sent to affinewarp"""

EPSILON = 1e-9
_FILE_WEIGHT_MODE_ALIASES = {
	"duration": "duration",
	"durations": "duration",
	"roi_duration": "duration",
	"roi_durations": "duration",
	"length": "duration",
	"uniform": "uniform",
	"equal": "uniform",
	"roi_count": "roi_count",
	"n_rois": "roi_count",
	"count": "roi_count",
}
_ROI_WEIGHT_MODE_ALIASES = {
	"duration": "duration",
	"durations": "duration",
	"length": "duration",
	"uniform": "uniform",
	"equal": "uniform",
}
_NORMALIZATION_MODE_ALIASES = {
	"none": "none",
	"off": "none",
	"disable": "none",
	"global": "global",
	"dataset": "global",
	"per_file": "per_file",
	"per-file": "per_file",
	"file": "per_file",
}
_NORMALIZATION_METHOD_ALIASES = {
	"mean_std": "mean_std",
	"meanstd": "mean_std",
	"mean": "mean_std",
	"std": "mean_std",
	"zscore": "mean_std",
	"z_score": "mean_std",
	"robust": "robust",
	"median_iqr": "robust",
	"median": "robust",
	"iqr": "robust",
}


def _default_num_workers(max_workers=8):
	cpu_count = os.cpu_count() or 1
	return max(0, min(max_workers, cpu_count - 1))


def _normalize_mode(value, default, aliases, name, allow_bool=False):
	if value is None:
		return default
	if allow_bool and isinstance(value, bool):
		value = "duration" if value else "uniform"
	if isinstance(value, str):
		key = value.strip().lower()
		if key in aliases:
			return aliases[key]
		valid = ", ".join(sorted(set(aliases.values())))
		raise ValueError(f"{name} must be one of: {valid}.")
	raise ValueError(f"{name} must be a string.")


def _normalize_choice(value, default, aliases, name, allow_bool=False,
	bool_map=None):
	if value is None:
		return default
	if allow_bool and isinstance(value, bool):
		if bool_map is None:
			bool_map = {True: "global", False: "none"}
		return bool_map[value]
	if isinstance(value, str):
		key = value.strip().lower()
		if key in aliases:
			return aliases[key]
		valid = ", ".join(sorted(set(aliases.values())))
		raise ValueError(f"{name} must be one of: {valid}.")
	raise ValueError(f"{name} must be a string or boolean.")


def _normalize_positive_int(value, default, name):
	if value is None:
		return int(default)
	try:
		value = int(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"{name} must be a positive integer.") from exc
	if value <= 0:
		raise ValueError(f"{name} must be a positive integer.")
	return value


def _normalize_bool(value, default, name):
	if value is None:
		return default
	if isinstance(value, bool):
		return value
	raise ValueError(f"{name} must be a boolean.")


def _normalize_weights(weights, label):
	weights = np.asarray(weights, dtype=float).reshape(-1)
	if np.any(~np.isfinite(weights)):
		raise ValueError(f"{label} weights contain non-finite values.")
	total = float(np.sum(weights))
	if total <= 0:
		raise ValueError(f"{label} weights must sum to a positive value.")
	return weights / total


def _seed_fixed_window_worker(worker_id):
	worker_info = torch.utils.data.get_worker_info()
	if worker_info is None:
		return
	seed = worker_info.seed % 2**32
	dataset = worker_info.dataset
	if hasattr(dataset, "seed"):
		dataset.seed(seed)


def _roi_has_data(filename):
	with open(filename, 'r') as handle:
		for line in handle:
			stripped = line.strip()
			if not stripped or stripped.startswith('#'):
				continue
			return True
	return False


def get_window_partition(audio_dirs, roi_dirs, split=0.8, shuffle=True, \
	exclude_empty_roi_files=True):
	"""
	Get a train/test split for fixed-duration shotgun VAE.

	Parameters
	----------
	audio_dirs : list of str
		Audio directories.
	roi_dirs : list of str
		ROI (segment) directories.
	split : float, optional
		Train/test split. Defaults to ``0.8``, indicating an 80/20 train/test
		split.
	shuffle : bool, optional
		Whether to shuffle at the audio file level. Defaults to ``True``.
	exclude_empty_roi_files : bool, optional
		Defaults to ``True``.

	Returns
	-------
	partition : dict
		Defines the test/train split. The keys ``'test'`` and ``'train'`` each
		map to a dictionary with keys ``'audio'`` and ``'rois'``, which both
		map to numpy arrays containing filenames.
	"""
	assert(split > 0.0 and split <= 1.0)
	# Collect filenames.
	audio_filenames, roi_filenames = [], []
	for audio_dir, roi_dir in zip(audio_dirs, roi_dirs):
		temp_wavs = _get_wavs_from_dir(audio_dir)
		temp_rois = [os.path.join(roi_dir, os.path.split(i)[-1][:-4]+'.txt') \
				for i in temp_wavs]
		if exclude_empty_roi_files:
			for i in reversed(range(len(temp_wavs))):
				if not _roi_has_data(temp_rois[i]):
					del temp_wavs[i]
					del temp_rois[i]
		audio_filenames += temp_wavs
		roi_filenames += temp_rois
	# Reproducibly shuffle.
	audio_filenames = np.array(audio_filenames)
	roi_filenames = np.array(roi_filenames)
	perm = np.argsort(audio_filenames)
	audio_filenames, roi_filenames = audio_filenames[perm], roi_filenames[perm]
	if shuffle:
		np.random.seed(42)
		perm = np.random.permutation(len(audio_filenames))
		audio_filenames = audio_filenames[perm]
		roi_filenames = roi_filenames[perm]
		np.random.seed(None)
	# Split.
	i = int(round(split * len(audio_filenames)))
	return { \
		'train': { \
			'audio': audio_filenames[:i], 'rois': roi_filenames[:i]}, \
		'test': { \
			'audio': audio_filenames[i:], 'rois': roi_filenames[i:]} \
		}


def get_fixed_window_data_loaders(partition, p, batch_size=64, \
	shuffle=(True, False), num_workers=None, min_spec_val=None, \
	min_audio_energy=None, spec_cache_dir=None, spec_cache=None, \
	audio_cache_size=0, pin_memory=None, persistent_workers=None, \
	prefetch_factor=2):
	"""
	Get DataLoaders for training and testing: fixed-duration shotgun VAE

	Parameters
	----------
	partition : dict
		Output of ``ava.models.window_vae_dataset.get_window_partition``.
	p : dict
		Preprocessing parameters. Must contain keys: ...
	batch_size : int, optional
		Defaults to ``64``.
	shuffle : tuple of bool, optional
		Whether to shuffle train and test sets, respectively. Defaults to
		``(True, False)``.
	num_workers : {int, None}, optional
		Number of CPU workers to feed data to the network. Defaults to
		``max(0, min(8, (os.cpu_count() or 1) - 1))`` when ``None``.
	min_spec_val : {float, None}, optional
		Used to disregard silence. If not `None`, spectrogram with a maximum
		value less than `min_spec_val` will be disregarded.
	min_audio_energy : {float, None}, optional
		Minimum mean-squared energy (after DC offset removal) required for a
		window to be considered. Windows below this threshold are resampled
		before spectrogram computation.
	spec_cache_dir : {str, None}, optional
		Directory for an on-disk spectrogram/amplitude cache. Defaults to
		``None`` (no on-disk cache).
	spec_cache : {dict, None}, optional
		Optional in-memory cache for spectrograms/amplitude traces. Defaults
		to ``None``.
	audio_cache_size : int, optional
		Maximum number of audio files to keep in an in-memory LRU cache.
		Defaults to ``0`` (no audio caching).
	pin_memory : {bool, None}, optional
		Whether to pin memory in returned batches. Defaults to ``True`` when
		CUDA is available.
	persistent_workers : {bool, None}, optional
		Whether to keep workers alive between epochs. Defaults to ``True``
		when ``num_workers > 0``.
	prefetch_factor : int, optional
		Number of batches to prefetch per worker. Defaults to ``2`` when
		``num_workers > 0``.

	Returns
	-------
	loaders : dict
		Maps the keys ``'train'`` and ``'test'`` to their respective
		DataLoaders.
	"""
	if num_workers is None:
		num_workers = _default_num_workers()
	num_workers = max(int(num_workers), 0)
	if pin_memory is None:
		pin_memory = torch.cuda.is_available()
	if persistent_workers is None:
		persistent_workers = num_workers > 0
	if num_workers == 0:
		persistent_workers = False
	loader_kwargs = {
		"num_workers": num_workers,
		"pin_memory": pin_memory,
		"worker_init_fn": _seed_fixed_window_worker,
	}
	if num_workers > 0:
		loader_kwargs["persistent_workers"] = persistent_workers
		if prefetch_factor is not None:
			loader_kwargs["prefetch_factor"] = prefetch_factor
	normalization_stats = p.get("normalization_stats", None)
	train_dataset = FixedWindowDataset(partition['train']['audio'], \
			partition['train']['rois'], p, transform=numpy_to_tensor, \
			min_spec_val=min_spec_val, min_audio_energy=min_audio_energy, \
			spec_cache_dir=spec_cache_dir, spec_cache=spec_cache, \
			audio_cache_size=audio_cache_size, \
			normalization_stats=normalization_stats)
	if normalization_stats is None and \
			train_dataset.normalization_mode == "global":
		normalization_stats = train_dataset.normalization_stats
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, \
			shuffle=shuffle[0], **loader_kwargs)
	if not partition['test']:
		return {'train':train_dataloader, 'test':None}
	test_dataset = FixedWindowDataset(partition['test']['audio'], \
			partition['test']['rois'], p, transform=numpy_to_tensor, \
			min_spec_val=min_spec_val, min_audio_energy=min_audio_energy, \
			spec_cache_dir=spec_cache_dir, spec_cache=spec_cache, \
			audio_cache_size=audio_cache_size, \
			normalization_stats=normalization_stats)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, \
			shuffle=shuffle[1], **loader_kwargs)
	return {'train':train_dataloader, 'test':test_dataloader}



class FixedWindowDataset(Dataset):

	def __init__(self, audio_filenames, roi_filenames, p, transform=None,
		dataset_length=2048, min_spec_val=None, min_audio_energy=None, \
		spec_cache_dir=None, spec_cache=None, audio_cache_size=0, \
		normalization_stats=None):
		"""
		Create a torch.utils.data.Dataset for chunks of animal vocalization.

		Parameters
		----------
		audio_filenames : list of str
			List of wav files.
		roi_filenames : list of str
			List of files containing animal vocalization times.
		p : dict
			Preprocessing parameters. Must include ``'window_length'`` and
			``'get_spec'``. Optional sampling keys: ``file_weight_mode``
			(``duration``, ``uniform``, ``roi_count``), ``file_weight_cap``
			(float), ``roi_weight_mode`` (``duration`` or ``uniform``),
			``sampling_seed`` (int), and ``log_window_indices`` (bool).
			Optional normalization keys: ``normalization_mode`` (``none``,
			``global``, ``per_file``), ``normalization_method`` (``mean_std``,
			``robust``), ``normalization_num_samples`` (int), and
			``normalization_seed`` (int).
		transform : {``None``, function}, optional
			Transformation to apply to each item. Defaults to ``None`` (no
			transformation).
		dataset_length : int, optional
			Arbitrary number that determines batch size. Defaults to ``2048``.
		min_spec_val : {float, None}, optional
			Used to disregard silence. If not `None`, spectrogram with a maximum
			value less than `min_spec_val` will be disregarded.
		min_audio_energy : {float, None}, optional
			Minimum mean-squared energy (after DC offset removal) required for a
			window to be considered. Windows below this threshold are resampled
			before spectrogram computation.
		spec_cache_dir : {str, None}, optional
			Directory for an on-disk spectrogram/amplitude cache. Defaults to
			``None`` (no on-disk cache).
		spec_cache : {dict, None}, optional
			Optional in-memory cache for spectrograms/amplitude traces. Defaults
			to ``None``.
		audio_cache_size : int, optional
			Maximum number of audio files to keep in an in-memory LRU cache.
			Defaults to ``0`` (no audio caching).
		normalization_stats : {dict, None}, optional
			Optional normalization statistics to reuse across datasets.
		"""
		sorted_pairs = sorted(zip(audio_filenames, roi_filenames), \
			key=lambda pair: pair[0])
		self.filenames = np.array([pair[0] for pair in sorted_pairs])
		self.fs = p.get('fs', None)
		self.roi_filenames = [pair[1] for pair in sorted_pairs]
		self.dataset_length = dataset_length
		self.min_spec_val = min_spec_val
		self.min_audio_energy = None
		if min_audio_energy is not None:
			try:
				min_audio_energy = float(min_audio_energy)
			except (TypeError, ValueError) as exc:
				raise ValueError(
					"min_audio_energy must be a non-negative number."
				) from exc
			if not np.isfinite(min_audio_energy) or min_audio_energy < 0:
				raise ValueError(
					"min_audio_energy must be a non-negative number."
				)
			self.min_audio_energy = min_audio_energy
		self.p = p
		window_length = self.p.get('window_length', None)
		if window_length is None:
			raise ValueError("FixedWindowDataset requires p['window_length'].")
		valid_filenames = []
		valid_roi_filenames = []
		self.rois = []
		self._roi_lengths = []
		for audio_fn, roi_fn in zip(self.filenames, self.roi_filenames):
			rois = np.loadtxt(roi_fn, ndmin=2)
			roi_lengths = np.diff(rois, axis=1).flatten()
			valid_mask = roi_lengths >= (window_length - EPSILON)
			if not np.any(valid_mask):
				warnings.warn(
					"No ROIs are long enough for window_length=" + \
						f"{window_length} in {roi_fn}; skipping {audio_fn}.",
					UserWarning
				)
				continue
			self.rois.append(rois[valid_mask])
			self._roi_lengths.append(roi_lengths[valid_mask])
			valid_filenames.append(audio_fn)
			valid_roi_filenames.append(roi_fn)
		self.filenames = np.array(valid_filenames)
		self.roi_filenames = np.array(valid_roi_filenames)
		if len(self.filenames) == 0:
			raise ValueError(
				"No ROIs are long enough for window_length=" + \
					f"{window_length}. Check ROI files or window_length."
			)
		file_weight_mode = _normalize_mode(
			self.p.get("file_weight_mode"),
			default="duration",
			aliases=_FILE_WEIGHT_MODE_ALIASES,
			name="file_weight_mode",
		)
		if file_weight_mode == "uniform":
			file_weights = np.ones(len(self._roi_lengths), dtype=float)
		elif file_weight_mode == "roi_count":
			file_weights = np.array(
				[len(lengths) for lengths in self._roi_lengths],
				dtype=float,
			)
		else:
			file_weights = np.array(
				[np.sum(lengths) for lengths in self._roi_lengths],
				dtype=float,
			)
		file_weight_cap = self.p.get("file_weight_cap", None)
		if file_weight_cap is not None:
			try:
				file_weight_cap = float(file_weight_cap)
			except (TypeError, ValueError) as exc:
				raise ValueError(
					"file_weight_cap must be a positive number."
				) from exc
			if file_weight_cap <= 0:
				raise ValueError("file_weight_cap must be positive.")
			file_weights = np.minimum(file_weights, file_weight_cap)
		self.file_weights = _normalize_weights(file_weights, "File")
		self.file_weight_mode = file_weight_mode
		self.file_weight_cap = file_weight_cap
		roi_weight_mode = _normalize_mode(
			self.p.get("roi_weight_mode"),
			default="duration",
			aliases=_ROI_WEIGHT_MODE_ALIASES,
			name="roi_weight_mode",
			allow_bool=True,
		)
		self.roi_weight_mode = roi_weight_mode
		self.roi_weights = []
		for lengths in self._roi_lengths:
			if roi_weight_mode == "uniform":
				roi_weights = np.ones(len(lengths), dtype=float)
			else:
				roi_total = float(np.sum(lengths))
				if roi_total <= 0:
					roi_weights = np.ones(len(lengths), dtype=float)
				else:
					roi_weights = np.array(lengths, dtype=float)
			self.roi_weights.append(_normalize_weights(roi_weights, "ROI"))
		self.transform = transform
		self.spec_cache_dir = spec_cache_dir
		self.spec_cache = spec_cache
		self.audio_cache_size = max(int(audio_cache_size), 0)
		self._audio_cache = OrderedDict()
		self._spec_cache_keys = {}
		if self.spec_cache_dir:
			os.makedirs(self.spec_cache_dir, exist_ok=True)
		self._rng = np.random.RandomState()
		sampling_seed = self.p.get("sampling_seed", None)
		if sampling_seed is not None:
			try:
				sampling_seed = int(sampling_seed)
			except (TypeError, ValueError) as exc:
				raise ValueError("sampling_seed must be an integer.") from exc
		self._sampling_seed = sampling_seed
		self._log_window_indices = _normalize_bool(
			self.p.get("log_window_indices"),
			default=self._sampling_seed is not None,
			name="log_window_indices",
		)
		self.window_log = {}
		if self._sampling_seed is not None or self._log_window_indices:
			self._epoch = multiprocessing.Value("i", 0)
		else:
			self._epoch = 0
		self.set_epoch(0)
		self._configure_normalization(normalization_stats)


	def seed(self, seed=None):
		if seed is None:
			self._rng = np.random.RandomState()
		else:
			self._rng = np.random.RandomState(int(seed))

	def set_epoch(self, epoch):
		try:
			epoch = int(epoch)
		except (TypeError, ValueError) as exc:
			raise ValueError("epoch must be an integer.") from exc
		if epoch < 0:
			raise ValueError("epoch must be non-negative.")
		if hasattr(self._epoch, "value"):
			self._epoch.value = epoch
		else:
			self._epoch = epoch
		if self._log_window_indices:
			self.window_log.setdefault(epoch, [])

	def _get_epoch(self):
		if hasattr(self._epoch, "value"):
			return int(self._epoch.value)
		return int(self._epoch)

	def _make_window_seed(self, epoch, index):
		payload = f"{self._sampling_seed}:{epoch}:{int(index)}"
		digest = hashlib.sha256(payload.encode("utf-8")).digest()
		return int.from_bytes(digest[:4], "little")

	def _record_window(self, epoch, dataset_index, file_index, roi_index, \
		onset, offset, seed=None):
		if not self._log_window_indices:
			return
		entry = {
			"dataset_index": int(dataset_index),
			"file_index": int(file_index),
			"roi_index": int(roi_index),
			"onset": float(onset),
			"offset": float(offset),
		}
		if seed is not None:
			entry["seed"] = int(seed)
		self.window_log.setdefault(int(epoch), []).append(entry)

	def get_window_log(self, epoch=None):
		if epoch is None:
			return self.window_log
		return self.window_log.get(int(epoch), [])

	def _configure_normalization(self, normalization_stats):
		self.normalization_mode = _normalize_choice(
			self.p.get("normalization_mode"),
			default="none",
			aliases=_NORMALIZATION_MODE_ALIASES,
			name="normalization_mode",
			allow_bool=True,
		)
		self.normalization_method = _normalize_choice(
			self.p.get("normalization_method"),
			default="mean_std",
			aliases=_NORMALIZATION_METHOD_ALIASES,
			name="normalization_method",
		)
		self.normalization_stats = None
		self._norm_center = None
		self._norm_scale = None
		if self.normalization_mode == "none":
			self.normalization_num_samples = None
			self.normalization_seed = None
			return
		self.normalization_num_samples = _normalize_positive_int(
			self.p.get("normalization_num_samples"),
			default=128,
			name="normalization_num_samples",
		)
		seed = self.p.get("normalization_seed", 0)
		try:
			self.normalization_seed = int(seed)
		except (TypeError, ValueError) as exc:
			raise ValueError("normalization_seed must be an integer.") from exc
		if normalization_stats is None:
			normalization_stats = self._compute_normalization_stats()
		self._set_normalization_stats(normalization_stats)

	def _set_normalization_stats(self, stats):
		self.normalization_stats = None
		self._norm_center = None
		self._norm_scale = None
		if stats is None:
			return
		if not isinstance(stats, dict):
			raise ValueError("normalization_stats must be a dict.")
		mode = stats.get("mode", self.normalization_mode)
		method = stats.get("method", self.normalization_method)
		if mode != self.normalization_mode:
			raise ValueError(
				"normalization_stats mode does not match normalization_mode."
			)
		if method != self.normalization_method:
			raise ValueError(
				"normalization_stats method does not match normalization_method."
			)
		center = stats.get("center", None)
		scale = stats.get("scale", None)
		if mode == "global":
			center = float(center)
			scale = float(scale)
		elif mode == "per_file":
			center = np.asarray(center, dtype=float).reshape(-1)
			scale = np.asarray(scale, dtype=float).reshape(-1)
			if center.shape[0] != len(self.filenames):
				raise ValueError(
					"normalization_stats center length does not match "
					"number of files."
				)
			if scale.shape[0] != len(self.filenames):
				raise ValueError(
					"normalization_stats scale length does not match "
					"number of files."
				)
		else:
			raise ValueError("normalization_stats mode must be global or per_file.")
		if np.any(~np.isfinite(scale)):
			raise ValueError("normalization_stats scale contains non-finite values.")
		if np.any(scale <= 0):
			raise ValueError("normalization_stats scale must be positive.")
		self.normalization_stats = {
			"mode": mode,
			"method": method,
			"center": center,
			"scale": scale,
		}
		self._norm_center = center
		self._norm_scale = scale

	def _compute_normalization_stats(self):
		seed = self.normalization_seed
		if self.normalization_mode == "global":
			rng = np.random.RandomState(seed)
			center, scale = self._compute_stats_for_samples(
				rng,
				self.normalization_num_samples,
				file_index=None,
			)
			return {
				"mode": "global",
				"method": self.normalization_method,
				"center": center,
				"scale": scale,
			}
		if self.normalization_mode == "per_file":
			centers = np.zeros(len(self.filenames), dtype=float)
			scales = np.zeros(len(self.filenames), dtype=float)
			for idx in range(len(self.filenames)):
				rng = np.random.RandomState(seed + idx)
				center, scale = self._compute_stats_for_samples(
					rng,
					self.normalization_num_samples,
					file_index=idx,
				)
				centers[idx] = center
				scales[idx] = scale
			return {
				"mode": "per_file",
				"method": self.normalization_method,
				"center": centers,
				"scale": scales,
			}
		raise ValueError("normalization_mode must be none, global, or per_file.")

	def _compute_stats_for_samples(self, rng, num_samples, file_index=None):
		if self.normalization_method == "robust":
			values = []
			for _ in range(num_samples):
				spec, _, _, _ = self._draw_window(
					rng,
					file_index=file_index,
					apply_normalization=False,
					max_attempts=100,
				)
				values.append(np.ravel(spec))
			if not values:
				raise ValueError("Failed to sample windows for normalization.")
			values = np.concatenate(values, axis=0)
			median = float(np.median(values))
			q75, q25 = np.percentile(values, [75, 25])
			scale = float(q75 - q25)
			if not np.isfinite(scale) or scale <= 0:
				scale = 1.0
			return median, scale
		total = 0.0
		total_sq = 0.0
		count = 0
		for _ in range(num_samples):
			spec, _, _, _ = self._draw_window(
				rng,
				file_index=file_index,
				apply_normalization=False,
				max_attempts=100,
			)
			values = np.asarray(spec, dtype=np.float64)
			total += float(values.sum())
			total_sq += float(np.square(values).sum())
			count += values.size
		if count == 0:
			raise ValueError("Failed to sample windows for normalization.")
		mean = total / count
		var = total_sq / count - mean ** 2
		if var < 0:
			var = 0.0
		std = float(np.sqrt(var))
		if not np.isfinite(std) or std <= 0:
			std = 1.0
		return float(mean), std

	def _apply_normalization(self, spec, file_index):
		if self._norm_center is None:
			return spec
		if self.normalization_mode == "global":
			center = self._norm_center
			scale = self._norm_scale
		else:
			center = self._norm_center[file_index]
			scale = self._norm_scale[file_index]
		return (spec - center) / scale


	def __len__(self):
		"""NOTE: length is arbitrary"""
		return self.dataset_length


	def _read_wav(self, filename):
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=WavFileWarning)
			try:
				fs, audio = wavfile.read(filename, mmap=True)
			except TypeError:
				fs, audio = wavfile.read(filename)
		return fs, audio


	def _get_audio(self, filename):
		if self.audio_cache_size > 0 and filename in self._audio_cache:
			self._audio_cache.move_to_end(filename)
			return self._audio_cache[filename]
		fs, audio = self._read_wav(filename)
		if self.fs is None:
			self.fs = fs
		elif fs != self.fs:
			warnings.warn(
				"Found inconsistent sample rate for " + filename + ".",
				UserWarning
			)
		if self.audio_cache_size > 0:
			self._audio_cache[filename] = audio
			self._audio_cache.move_to_end(filename)
			while len(self._audio_cache) > self.audio_cache_size:
				self._audio_cache.popitem(last=False)
		return audio


	def _window_energy(self, audio, onset, offset):
		start = int(round(onset * self.fs))
		stop = int(round(offset * self.fs))
		if stop <= 0 or start >= len(audio):
			return 0.0
		segment = audio[max(0, start):min(len(audio), stop)]
		if segment.size == 0:
			return 0.0
		segment = segment.astype(np.float32)
		segment = segment - np.mean(segment)
		return float(np.mean(np.square(segment)))


	def _spec_cache_params(self):
		spec_keys = [
			'nperseg',
			'noverlap',
			'min_freq',
			'max_freq',
			'spec_min_val',
			'spec_max_val',
			'num_freq_bins',
			'num_time_bins',
			'mel',
			'time_stretch',
			'max_dur',
			'within_syll_normalize',
			'normalize_quantile',
			'window_length',
		]
		params = {key: self.p.get(key, None) for key in spec_keys}
		get_spec = self.p.get('get_spec', None)
		params['get_spec'] = getattr(get_spec, '__name__', repr(get_spec))
		return params


	def _make_spec_cache_key(self, filename, onset, offset, shoulder):
		if filename in self._spec_cache_keys:
			base_key = self._spec_cache_keys[filename]
		else:
			try:
				file_stats = {
					'path': os.path.abspath(filename),
					'mtime': os.path.getmtime(filename),
					'size': os.path.getsize(filename),
				}
			except OSError:
				file_stats = {
					'path': os.path.abspath(filename),
					'mtime': None,
					'size': None,
				}
			base_payload = {
				'file': file_stats,
				'params': self._spec_cache_params(),
			}
			base_json = json.dumps(base_payload, sort_keys=True, default=str)
			base_key = hashlib.sha256(base_json.encode('utf-8')).hexdigest()
			self._spec_cache_keys[filename] = base_key
		window_payload = {
			'base': base_key,
			'onset': float(onset),
			'offset': float(offset),
			'shoulder': float(shoulder),
			'num_time_bins': int(self.p['num_time_bins']),
		}
		window_json = json.dumps(window_payload, sort_keys=True, default=str)
		return hashlib.sha256(window_json.encode('utf-8')).hexdigest()


	def _spec_cache_get(self, cache_key):
		if self.spec_cache is not None and cache_key in self.spec_cache:
			return self.spec_cache[cache_key]
		if self.spec_cache_dir:
			cache_path = os.path.join(self.spec_cache_dir, cache_key + '.npz')
			if os.path.exists(cache_path):
				with np.load(cache_path) as data:
					return {
						'spec': data['spec'],
						'amp': data['amp'],
					}
		return None


	def _spec_cache_set(self, cache_key, spec, amp):
		if self.spec_cache is not None:
			self.spec_cache[cache_key] = {'spec': spec, 'amp': amp}
		if self.spec_cache_dir:
			cache_path = os.path.join(self.spec_cache_dir, cache_key + '.npz')
			if not os.path.exists(cache_path):
				with tempfile.NamedTemporaryFile(
					suffix='.npz',
					dir=self.spec_cache_dir,
					delete=False
				) as tmp:
					tmp_path = tmp.name
				try:
					np.savez_compressed(tmp_path, spec=spec, amp=amp)
					os.replace(tmp_path, cache_path)
				except OSError:
					if os.path.exists(tmp_path):
						os.remove(tmp_path)

	def _draw_window(self, rng, shoulder=0.05, file_index=None, \
		apply_normalization=True, max_attempts=None, \
		return_roi_index=False):
		attempts = 0
		fixed_file = file_index is not None
		while True:
			if max_attempts is not None and attempts >= max_attempts:
				raise ValueError(
					"Unable to sample a valid window for normalization."
				)
			attempts += 1
			if fixed_file:
				current_file_index = int(file_index)
			else:
				current_file_index = rng.choice(
					len(self.filenames),
					p=self.file_weights
				)
			load_filename = self.filenames[current_file_index]
			roi_index = rng.choice(
				len(self.roi_weights[current_file_index]),
				p=self.roi_weights[current_file_index]
			)
			roi = self.rois[current_file_index][roi_index]
			onset = roi[0] + (roi[1] - roi[0] - self.p['window_length']) \
				* rng.rand()
			offset = onset + self.p['window_length']
			audio = None
			if self.min_audio_energy is not None:
				audio = self._get_audio(load_filename)
				if self._window_energy(audio, onset, offset) \
						< self.min_audio_energy:
					continue
			target_times = np.linspace(onset, offset, self.p['num_time_bins'])
			cache_key = None
			spec = None
			if self.spec_cache_dir or self.spec_cache is not None:
				cache_key = self._make_spec_cache_key(
					load_filename,
					onset,
					offset,
					shoulder
				)
				cached = self._spec_cache_get(cache_key)
				if cached is not None:
					spec = cached['spec']
			if spec is None:
				if audio is None:
					audio = self._get_audio(load_filename)
				spec, flag = self.p['get_spec'](max(0.0, onset-shoulder), \
						offset+shoulder, audio, self.p, fs=self.fs, \
						target_times=target_times)
				if cache_key is not None:
					amp = np.sum(spec, axis=0, keepdims=True).T
					self._spec_cache_set(cache_key, spec, amp)
			else:
				flag = True
			if not flag:
				continue
			if self.min_spec_val is not None and \
					np.max(spec) < self.min_spec_val:
				continue
			if apply_normalization:
				spec = self._apply_normalization(spec, current_file_index)
			if return_roi_index:
				return spec, current_file_index, onset, offset, roi_index
			return spec, current_file_index, onset, offset


	def __getitem__(self, index, seed=None, shoulder=0.05, \
		return_seg_info=False):
		"""
		Get spectrograms.

		Parameters
		----------
		index :
		seed :
			If provided, use a deterministic RNG for this call. When
			``sampling_seed`` is set in ``p``, sampling is deterministic per
			index and epoch (see ``set_epoch``).
		shoulder :
		return_seg_info :

		Returns
		-------
		specs :
		file_indices :
		onsets :
		offsets :
		"""
		specs, file_indices, onsets, offsets = [], [], [], []
		single_index = False
		try:
			iterator = iter(index)
		except TypeError:
			index = [index]
			single_index = True
		epoch = self._get_epoch()
		base_rng = None
		if seed is not None:
			base_rng = np.random.RandomState(seed)
		for i in index:
			if seed is not None:
				rng = base_rng
				window_seed = None
			elif self._sampling_seed is not None:
				window_seed = self._make_window_seed(epoch, i)
				rng = np.random.RandomState(window_seed)
			else:
				rng = self._rng
				window_seed = None
			if self._log_window_indices:
				spec, file_index, onset, offset, roi_index = self._draw_window(
					rng,
					shoulder=shoulder,
					apply_normalization=True,
					return_roi_index=True,
				)
				self._record_window(
					epoch,
					i,
					file_index,
					roi_index,
					onset,
					offset,
					seed=window_seed,
				)
			else:
				spec, file_index, onset, offset = self._draw_window(
					rng,
					shoulder=shoulder,
					apply_normalization=True,
				)
			if self.transform:
				spec = self.transform(spec)
			specs.append(spec)
			file_indices.append(file_index)
			onsets.append(onset)
			offsets.append(offset)
		if return_seg_info:
			if single_index:
				return specs[0], file_indices[0], onsets[0], offsets[0]
			return specs, file_indices, onsets, offsets
		if single_index:
			return specs[0]
		return specs


	def write_hdf5_files(self, save_dir, num_files=500, sylls_per_file=100):
		"""
		Write hdf5 files containing spectrograms of random audio chunks.

		TO DO
		-----
		* Write to multiple directories.

		Note
		----
	 	* This should be consistent with
		  `ava.preprocessing.preprocess.process_sylls`.

		Parameters
		----------
		save_dir : str
			Directory to save hdf5s in.
		num_files : int, optional
			Number of files to save. Defaults to ``500``.
		sylls_per_file : int, optional
			Number of syllables in each file. Defaults to ``100``.
		"""
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)
		for write_file_num in range(num_files):
			specs, file_indices, onsets, offsets = \
					self.__getitem__(np.arange(sylls_per_file), \
					seed=write_file_num, return_seg_info=True)
			specs = np.array([spec.detach().numpy() for spec in specs])
			filenames = np.array([self.filenames[i] for i in file_indices])
			fn = "syllables_" + str(write_file_num).zfill(4) + '.hdf5'
			fn = os.path.join(save_dir, fn)
			with h5py.File(fn, "w") as f:
				f.create_dataset('specs', data=specs)
				f.create_dataset('onsets', data=np.array(onsets))
				f.create_dataset('offsets', data=np.array(offsets))
				f.create_dataset('audio_filenames', data=filenames.astype('S'))



def get_warped_window_data_loaders(audio_dirs, p, batch_size=64, num_workers=4,\
	load_warp=False, warp_fn=None, warp_params={}, warp_type='spectrogram'):
	"""
	Get DataLoaders for training and testing: warped shotgun VAE

	Warning
	-------
	* Audio files must all be the same duration! You can use
	  `segmenting.utils.write_segments_to_audio` to extract audio from song
	  segments, writing them as separate ``.wav`` files.

	TO DO
	-----
	* Add a train/test split!

	Parameters
	----------
	audio_dirs : list of str
		Audio directories.
	p : dict
		Preprocessing parameters. Must contain keys: ``'window_length'``,
		``'nperseg'``, ``'noverlap'``, ``'min_freq'``, ``'max_freq'``,
		``'spec_min_val'``, and ``'spec_max_val'``.
	batch_size : int, optional
		DataLoader batch size. Defaults to ``64``.
	num_workers : int, optional
		Number of CPU workers to retrieve data for the model. Defaults to ``4``.
	load_warp : bool, optional
		Whether to load a previously saved time warping result. Defaults to
		``False``.
	warp_fn : {str, None}, optional
		Where the x-knots and y-knots should be saved and loaded. Defaults to
		``None``.
	warp_params : dict, optional
		Parameters passed to affinewarp. Defaults to ``{}``.
	warp_type : {``'amplitude'``, ``'spectrogram'``, ``'null'``}, optional
		Whether to time-warp using ampltidue traces, full spectrograms, or not
		warp at all. Defaults to ``'spectrogram'``.

	Returns
	-------
	loaders : dict
		Maps the keys ``'train'`` and ``'test'`` to their respective
		DataLoaders.
	"""
	assert type(p) == type({})
	assert warp_type in ['amplitude', 'spectrogram', 'null']
	# Collect audio filenames.
	audio_fns = []
	for audio_dir in audio_dirs:
		audio_fns += _get_wavs_from_dir(audio_dir)
	# Make the Dataset and DataLoader.
	dataset = WarpedWindowDataset(audio_fns, p, \
		transform=numpy_to_tensor, load_warp=load_warp, warp_fn=warp_fn, \
		warp_params=warp_params, warp_type=warp_type)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, \
		num_workers=num_workers)
	return {'train': dataloader, 'test': dataloader}



class WarpedWindowDataset(Dataset):

	def __init__(self, audio_filenames, p, transform=None, dataset_length=2048,\
		load_warp=False, save_warp=True, start_q=-0.1, stop_q=1.1, \
		warp_fn=None, warp_params={}, warp_type='spectrogram'):
		"""
		Dataset for time-warped chunks of animal vocalization

		TO DO
		-----
		* Use `affinewarp` functions instead of direct references to knots.

		Parameters
		----------
		audio_filenames : list of strings
			List of .wav files.
		p : dict
			Preprocessing parameters. Must contain keys: ``'window_length'``,
			``'nperseg'``, ``'noverlap'``, ``'min_freq'``, ``'max_freq'``,
			``'spec_min_val'``, and ``'spec_max_val'``.
		transform : {None, function}, optional
			Transformation to apply to each item. Defaults to ``None`` (no
			transformation).
		dataset_length : int, optional
			Defaults to ``2048``. This is an arbitrary number that determines
			how many batches make up an epoch.
		load_warp : bool, optional
			Whether to load the results of a previous warp. Defaults to
			``False``.
		save_warp : bool, optional
			Whether to save the results of the warp. Defaults to ``True``.
		start_q : float, optional
			Start quantile. Defaults to ``-0.1``.
		stop_q : float, optional
			Stop quantile. Defaults to ``1.1``.
		warp_fn : {None, str}, optional
			Where to save the x knots and y knots of the warp. If ``None``, then
			nothing will be saved or loaded. Defaults to ``None``.
		warp_params : dict, optional
			Parameters passed to affinewarp. Defaults to ``{}``.
		warp_type : {``'amplitude'``, ``'spectrogram'``, ``'null'``}, optional
			Whether to time-warp using ampltidue traces, full spectrograms, or
			not warp at all. Defaults to ``'spectrogram'``.
		"""
		assert type(p) == type({})
		assert warp_type in ['amplitude', 'spectrogram', 'null']
		self.audio_filenames = sorted(audio_filenames)
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=WavFileWarning)
			self.audio = [wavfile.read(fn)[1] for fn in self.audio_filenames]
			self.fs = wavfile.read(self.audio_filenames[0])[0]
		self.dataset_length = dataset_length
		self.p = p
		self.transform = transform
		self.start_q = start_q
		self.stop_q = stop_q
		self.warp_fn = warp_fn
		self.warp_params = {**DEFAULT_WARP_PARAMS, **warp_params}
		self._compute_warp(load_warp=load_warp, save_warp=save_warp, \
				warp_type=warp_type)
		self.window_frac = self.p['window_length'] / self.template_dur


	def __len__(self):
		"""NOTE: length is arbitrary."""
		return self.dataset_length


	def write_hdf5_files(self, save_dir, num_files=400, sylls_per_file=100):
		"""
		Write hdf5 files containing spectrograms of random audio chunks.

		Note
		----
	 	This should be consistent with
		``ava.preprocessing.preprocess.process_sylls``.

		TO DO
		-----
		* Add the option to also write segments. This could be useful for noise
		  removal.

		Parameters
		----------
		save_dir : str
			Where to write.
		num_files : int, optional
			Number of files to write. Defaults to `400`.
		sylls_per_file : int, optional
			Number of spectrograms to write per file. Defaults to `100`.
		"""
		if save_dir != '' and not os.path.exists(save_dir):
			os.mkdir(save_dir)
		for write_file_num in range(num_files):
			specs = self.__getitem__(np.arange(sylls_per_file),
					seed=write_file_num)
			specs = np.array([spec.detach().numpy() for spec in specs])
			fn = "sylls_" + str(write_file_num).zfill(4) + '.hdf5'
			fn = os.path.join(save_dir, fn)
			with h5py.File(fn, "w") as f:
				f.create_dataset('specs', data=specs)


	def _get_unwarped_times(self, y_vals, index):
		"""
		Convert warped quantile times in [0,1] to real quantile times.

		Assumes y_vals is sorted.

		In affinewarp, x-values are empirical times, stored as quantiles from
		0 to 1, and y-values are template times. Here, we're given template
		times and converting to empirical times. In other words we're
		considering measured times as ``unwarped'' and aligned times as
		``warped''.
		"""
		x_knots, y_knots = self.x_knots[index], self.y_knots[index]
		interp = interp1d(y_knots, x_knots, bounds_error=False, \
				fill_value='extrapolate', assume_sorted=True)
		x_vals = interp(y_vals)
		return x_vals


	def _compute_warp(self, load_warp=False, save_warp=True, \
		warp_type='spectrogram'):
		"""
		Jointly warp all the song motifs.

		Warping is performed on spectrograms if ``warp_type == 'spectrogram'``.
		Otherwise, if ``warp_type == 'amplitude'``, warping is performed on
		spectrograms summed over the frequency dimension.
		"""
		if PiecewiseWarping is None:
			raise ModuleNotFoundError(
				"affinewarp is required for WarpedWindowDataset warping."
			)
		if save_warp:
			assert self.warp_fn is not None, "``warp_fn`` must be specified " +\
					"to save warps!"
		# If it's a null warp, make it and return.
		if warp_type == 'null':
			knots = np.zeros((len(self.audio),2))
			knots[:,1] = 1.0
			self.x_knots = knots
			self.y_knots = np.copy(knots)
			_, _, template_dur = _get_specs_and_amplitude_traces(self.audio,\
					self.fs, self.p)
			self.template_dur = template_dur
			print("Made null warp.")
			if save_warp:
				print("Saving warp to:", self.warp_fn)
				to_save = {
					'x_knots' : self.x_knots,
					'y_knots' : self.y_knots,
					'template_dur' : self.template_dur,
					'audio_filenames' : self.audio_filenames,
					'warp_params': self.warp_params,
				}
				np.save(self.warp_fn, to_save)
			return
		# Load warps if we can.
		if load_warp:
			if self.warp_fn is None:
				warnings.warn(
					"Tried to load warps, but ``warp_fns`` is None.",
					UserWarning
				)
			else:
				try:
					data = np.load(self.warp_fn, allow_pickle=True).item()
					self.x_knots = data['x_knots']
					self.y_knots = data['y_knots']
					self.template_dur = data['template_dur']
					temp_fns = data['audio_filenames']
					assert np.all(temp_fns[:-1] <= temp_fns[1:]), "Filenames "+\
							"in " + self.warp_fn + " are not sorted!"
					assert len(temp_fns) >= len(self.audio_filenames)
					if len(temp_fns) == len(self.audio_filenames):
						# If the saved filenames and the passed filenames have
						# the same length, make sure they match.
						assert np.array_equal(temp_fns, self.audio_filenames), \
								"Input filenames do not match saved filenames!"
					else:
						# Otherwise, make sure the passed filenames are a subset
						# of the saved filenames and keep track of the correct
						# indices.
						unique_fns = np.unique(self.audio_filenames)
						assert len(self.audio_filenames) == len(unique_fns)
						perm = np.zeros(len(self.audio_filenames), dtype='int')
						for i in range(len(self.audio_filenames)):
							assert self.audio_filenames[i] in temp_fns, \
									"Could not find filename " + \
									self.audio_filenames[i] + " in saved warps!"
							index = temp_fns.index(self.audio_filenames[i])
							perm[i] = index
						self.x_knots = self.x_knots[perm]
						self.y_knots = self.y_knots[perm]
					if type(self.audio_filenames) == type(np.array([])):
						self.audio_filenames = self.audio_filenames.tolist()
					self.warp_params = data['warp_params']
					return
				except IOError:
					warnings.warn(
						"Can't load warps from: "+str(self.warp_fn),
						UserWarning
					)
		# Otherwise, first make the spectrograms.
		specs, amps, template_dur = _get_specs_and_amplitude_traces(self.audio,\
				self.fs, self.p)
		self.template_dur = template_dur
		# Then warp.
		model = PiecewiseWarping(**self.warp_params)
		if warp_type == 'amplitude':
			print("Computing amplitude warp:", amps.shape)
			model.fit(amps, iterations=50, warp_iterations=200)
		elif warp_type == 'spectrogram':
			print("Computing spectrogram warp:", specs.shape)
			model.fit(specs, iterations=50, warp_iterations=200)
		else:
			raise NotImplementedError
		# Save the warps.
		self.x_knots = model.x_knots
		self.y_knots = model.y_knots
		if save_warp:
			print("Saving warp to:", self.warp_fn)
			to_save = {
				'x_knots' : self.x_knots,
				'y_knots' : self.y_knots,
				'template_dur' : self.template_dur,
				'audio_filenames' : self.audio_filenames,
				'amplitude_traces': amps,
				'warp_params': self.warp_params,
			}
			np.save(self.warp_fn, to_save)


	def __getitem__(self, index, seed=None):
		"""
		Return a random window of birdsong.

		Parameters
		----------
		index : {int, list of int}
			Determines the number of spectrograms to return. If an int is
			passed, a single spectrogram is returned. If a list is passed,
			``len(index)`` spectrograms are returned. Elements (ints)
			themselves are ignored.
		seed : {None, int}, optional
			Random seed

		Returns
		-------
		spec : {numpy.ndarray, list of numpy.ndarray}
			Spectrograms
		"""
		result = []
		single_index = False
		try:
			iterator = iter(index)
		except TypeError:
			index = [index]
			single_index = True
		np.random.seed(seed)
		for i in index:
			while True:
				# First find the file, then the ROI.
				file_index = np.random.randint(len(self.audio))
				# Then choose a chunk of audio uniformly at random.
				start_t = self.start_q + np.random.rand() * \
						(self.stop_q - self.start_q - self.window_frac)
				stop_t = start_t + self.window_frac
				t_vals = np.linspace(start_t, stop_t, self.p['num_time_bins'])
				# Inverse warp.
				target_ts = self._get_unwarped_times(t_vals, file_index)
				target_ts *= self.template_dur
				# Then make a spectrogram.
				spec, flag = self.p['get_spec'](0.0, self.template_dur, \
						self.audio[file_index], self.p, fs=self.fs, \
						max_dur=None, target_times=target_ts)
				assert flag
				if self.transform:
					spec = self.transform(spec)
				result.append(spec)
				break
		np.random.seed(None)
		if single_index:
			return result[0]
		return result


	def get_specific_item(self, query_filename, quantile):
		"""
		Return a specific window of birdsong as a Numpy array.

		Parameters
		----------
		query_filename : str
			Audio filename.
		quantile : float
			0 <= ``quantile`` <= 1

		Returns
		-------
		spec : numpy.ndarray
			Spectrogram.
		"""
		file_index = self.audio_filenames.index(query_filename)
		start_t = self.start_q + quantile * \
				(self.stop_q - self.start_q - self.window_frac)
		stop_t = start_t + self.window_frac
		t_vals = np.linspace(start_t, stop_t, self.p['num_time_bins'])
		# Inverse warp.
		target_ts = self._get_unwarped_times(t_vals, file_index)
		target_ts *= self.template_dur
		# Then make a spectrogram.
		spec, flag = self.p['get_spec'](0.0, self.template_dur, \
				self.audio[file_index], self.p, fs=self.fs, \
				max_dur=None, target_times=target_ts)
		assert flag
		return spec


	def get_whole_warped_spectrogram(self, query_filename, time_bins=128):
		"""
		Get an entire warped song motif.

		Parameters
		----------
		query_filename : str
			Which audio file to use.
		time_bins : int, optional
			Number of time bins.

		Returns
		-------
		spec : numpy.ndarray
			Spectrogram.
		"""
		file_index = self.audio_filenames.index(query_filename)
		t_vals = np.linspace(self.start_q, self.stop_q, time_bins)
		# Inverse warp.
		target_ts = self._get_unwarped_times(t_vals, file_index)
		target_ts *= self.template_dur
		# Then make a spectrogram.
		spec, flag = self.p['get_spec'](0.0, self.template_dur, \
				self.audio[file_index], self.p, fs=self.fs, \
				max_dur=None, target_times=target_ts)
		assert flag
		return spec



if __name__ == '__main__':
	pass


###
