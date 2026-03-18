Training
========

This section describes how to train the VAE.

Training on Syllables
#####################

By this point, we should have a list of directories containing preprocessed
syllable spectrograms. Our first step is to define PyTorch Dataloader objects
that are responsible for shipping data to and from the GPU. We want one for a
training set, and one for a test set.

.. code:: Python3

	spec_dirs = [...] # directories containing saved spectrograms (hdf5s)
	split = 0.8 # 80/20 train/test split

	# Construct a random train/test partition.
	from ava.models.vae_dataset import get_syllable_partition
	partition = get_syllable_partition(spec_dirs, split)

	# Make Dataloaders.
	from ava.models.vae_dataset import get_syllable_data_loaders
	loaders = get_syllable_data_loaders(partition)



Now we're ready to train the VAE.


.. code:: Python3

	# Construct network and train with Lightning.
	from ava.models.lightning_vae import train_vae
	save_dir = 'model/parameters/should/be/saved/here/'
	model, trainer = train_vae(loaders, save_dir=save_dir, epochs=101)



This should periodically save the model, print train and test loss, and write
a file in :code:`save_dir` called :code:`reconstruction.pdf` which displays
several spectrograms and their reconstructions.

Training Dashboard
##################

The Lightning training loop now also writes two dashboard artifacts into
:code:`save_dir`:

- :code:`training_dashboard.json`: structured status + metric history
- :code:`training_dashboard.html`: self-contained progress page

The HTML page refreshes automatically while the run is active and is intended
to be easy to inspect from a remote machine. Because it is plain static HTML,
you can serve it with any simple file server, copy it off a cloud instance, or
sync it to object storage from an AWS job directory without running TensorBoard.

When validation runs are enabled, the dashboard also includes a sampled latent
geometry panel for the validation set:

- an effective dimensionality estimate of the validation latent cloud
- a 2D PCA projection of sampled validation windows
- numbered validation spectrogram thumbnails linked to points in that projection

This is intentionally window-level rather than whole-audio-level, because the
training/validation loaders operate on spectrogram windows.

For older runs that only have :code:`lightning_logs/`, you can backfill the
same dashboard view from the saved TensorBoard event files:

.. code:: bash

	python scripts/render_training_dashboard.py \
		--run-dir path/to/training_run \
		--status completed

If you want a quick remote preview over SSH, one straightforward option is:

.. code:: bash

	cd path/to/training_run
	python -m http.server 8000

Then port-forward or expose that directory and open
:code:`training_dashboard.html` in a browser.

Residual blocks are the current default. To be explicit, pass
:code:`conv_arch="residual"` to :code:`train_vae` or :class:`ava.models.vae.VAE`,
or set :code:`training.conv_arch: residual` in fixed-window configs. The legacy
label :code:`plain` is still accepted as an alias for backward compatibility
(it uses the residual blocks).

You may also want to continue training a previously saved model:


.. code:: Python3

	# Make an untrained model.
	from ava.models.vae import VAE
	vae = VAE(save_dir=save_dir)

	# Load saved state.
	model_checkpoint = 'path/to/checkpoint_100.tar'
	vae.load_state(model_checkpoint)

	# Train another few epochs.
	from ava.models.lightning_vae import train_vae
	model, trainer = train_vae(loaders, vae=vae, epochs=51)




Checkpoint Compatibility (Decoder Type)
#######################################

Recent releases default to the upsample-based decoder. Older checkpoints were
saved with the legacy ConvTranspose2d decoder. The decoder type is stored in
the checkpoint and must match the model you instantiate.

If you see a decoder mismatch error while calling :code:`load_state`, build the
VAE with :code:`decoder_type="convtranspose"` (or set
:code:`training.decoder_type: convtranspose` in fixed-window configs) to load
legacy checkpoints. There is no weight-migration between decoder types, so
switching decoders requires retraining or exporting a new checkpoint with the
target decoder.

Another VAE parameter worth knowing about is :code:`model_precision`, which
controls the reconstruction/regularization tradeoff of the model. Very large
positive values will make the VAE behave more like a deterministic autoencoder,
encouraging better reconstructions. Small positive values produce
better-behaved latents, but with poorer reconstructions.
The default value is 10.  
If you want the observation scale to be learned instead, pass
:code:`learn_observation_scale=True`. In that case, :code:`model_precision`
sets the initial value.


Shotgun VAE Training
####################

Training the shotgun VAE is pretty similar to training the syllable VAE, the
main difference being the Dataloader we feed it: instead of making spectrograms
beforehand, we make them during training.

.. code:: Python3

	# Define parameters and directories.
	params = {...}
	audio_dirs = [...]
	roi_dirs = [...] # same format as syllable segments

	# Make a Dataloader.
	from ava.models.shotgun_vae_dataset import get_shotgun_partition, \
			get_fixed_shotgun_data_loaders
	split = 0.8 # 80/20 train/test split
	partition = get_shotgun_partition(audio_dirs, roi_dirs, split)
	loaders = get_fixed_shotgun_data_loaders(partition, params)


Then training is the same as before:

.. code:: Python3

	# Train.
	from ava.models.lightning_vae import train_vae
	save_dir = 'model/parameters/should/be/saved/here/'
	model, trainer = train_vae(loaders, save_dir=save_dir, epochs=101)


Note that we define segments for the shotgun VAE in :code:`roi_dirs`. These
should have the same format as syllable segments, but should cover longer
periods of vocalization.


Latent Invariance Evaluation
############################

AVA includes a CLI for evaluating latent invariance and self-retrieval on
paired views:

.. code:: bash

	python scripts/evaluate_latent_metrics.py \
		--config path/to/fixed_window_config.yaml \
		--checkpoint path/to/checkpoint_100.tar \
		--audio-dir /path/to/wavs \
		--roi-dir /path/to/rois \
		--split test \
		--max-samples 20000 \
		--batch-size 64 \
		--device cpu

The output JSON includes:

- :code:`latent_invariance.mean`, :code:`median`, :code:`std`,
  :code:`p05`, :code:`p95`: L2 distances between paired latent means
  (lower means stronger invariance).
- :code:`self_retrieval.top1`, :code:`top5`, :code:`mean_rank`,
  :code:`median_rank`: how often each augmented view retrieves its paired
  base view by nearest neighbor (higher top-k and lower ranks are better).
- :code:`num_pairs`: number of paired windows evaluated.

Topology readiness heuristic: invariance distances should be stable across
runs and small relative to the latent scale, while self-retrieval top-1/top-5
should be high and mean rank should stay near 1. If invariance distances
collapse to ~0 while reconstructions degrade, reduce augmentation strength or
increase reconstruction weight.

Recommended baseline augmentation config for birdsong:

.. code:: yaml

	augmentations:
	  enabled: true
	  seed: 123
	  amplitude_scale: [0.9, 1.1]
	  noise_std: 0.02
	  time_shift_max_bins: 2
	  freq_shift_max_bins: 2
	  time_mask_max_bins: 4
	  time_mask_count: 1
	  freq_mask_max_bins: 3
	  freq_mask_count: 1


Warped Shotgun VAE Training
###########################

TO DO

Mode Collapse
#############

One possible issue during training is known as mode collapse or posterior
collapse. This happens when
the VAE's tendency to regularize overwhelms its ability to reconstruct
spectrograms, and is the tendency of the VAE to ignore its input so that each
reconstruction is simply the mean spectrogram. There are two ways to deal with
this in AVA. First, we can increase the contrast of the spectrograms by
decreasing the range between :code:`'spec_min_val'` and :code:`'spec_max_val'`
in the preprocessing step. Second, we can increase the model precision in the
training step to strike a different regularization/reconstruction tradeoff:

.. code:: Python3

	model = VAE(model_precision=20.0) # default is 10.0
