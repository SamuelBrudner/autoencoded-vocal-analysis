"""
Minimal working example for shotgun VAE using unwarped birdsong.

0) Define directories and parameters.
1) Tune preprocessing parameters.
2) Train the VAE.
3) Plot.
4) The world is your oyster.

"""

import os

from ava.data.data_container import DataContainer
from ava.models.lightning_vae import train_vae
from ava.models.fixed_window_config import FixedWindowExperimentConfig
from ava.models.shotgun_vae_dataset import get_shotgun_partition, \
	get_fixed_shotgun_data_loaders
from ava.preprocessing.preprocess import tune_window_preprocessing_params


#########################################
# 0) Define directories and parameters. #
#########################################
config_path = os.path.join(
	os.path.dirname(__file__),
	"configs",
	"fixed_window_finch.yaml",
)
config = FixedWindowExperimentConfig.from_yaml(config_path)
params = config.preprocess.to_params()
data_config = config.data
train_config = config.training
root = '/path/to/directory/'
audio_dirs = [os.path.join(root, 'audio')]
roi_dirs = [os.path.join(root, 'segs')]
spec_dirs = [os.path.join(root, 'h5s')]
proj_dirs = [os.path.join(root, 'proj')]
model_filename = os.path.join(root, 'checkpoint_100.tar')
plots_dir = root


dc = DataContainer(projection_dirs=proj_dirs, audio_dirs=audio_dirs, \
	spec_dirs=spec_dirs, plots_dir=root, model_filename=model_filename)


#####################################
# 1) Tune preprocessing parameters. #
#####################################
params = tune_window_preprocessing_params(audio_dirs, params)


###################################################
# 2) Train a generative model on these syllables. #
###################################################
partition = get_shotgun_partition(audio_dirs, roi_dirs, 1)
partition['test'] = partition['train']
loader_kwargs = data_config.to_loader_kwargs()
num_workers = data_config.num_workers
if num_workers is None:
	num_workers = min(7, (os.cpu_count() or 1) - 1)
	num_workers = max(num_workers, 0)
loader_kwargs["num_workers"] = num_workers
use_pairs = train_config.invariance_weight > 0
loaders = get_fixed_shotgun_data_loaders(
	partition,
	params,
	return_pair=use_pairs,
	pair_with_original=use_pairs,
	**loader_kwargs,
)
loaders['test'] = loaders['train']
train_kwargs = train_config.to_train_kwargs()
model, trainer = train_vae(loaders, save_dir=root, **train_kwargs)


############
# 3) Plot. #
############
from ava.plotting.tooltip_plot import tooltip_plot_DC
from ava.plotting.latent_projection import latent_projection_plot_DC

# Write random spectrograms into a single directory.
loaders['test'].dataset.write_hdf5_files(spec_dirs[0], num_files=1000)

# Redefine the DataContainer so it only looks in that single directory.
temp_dc = DataContainer(projection_dirs=proj_dirs[:1], \
	audio_dirs=audio_dirs[:1], spec_dirs=spec_dirs[:1], plots_dir=root, \
	model_filename=model_filename)

latent_projection_plot_DC(temp_dc, alpha=0.25, s=0.5)
tooltip_plot_DC(temp_dc, num_imgs=2000)


################################
# 4) The world is your oyster. #
################################
latent = dc.request('latent_means')
pass



if __name__ == '__main__':
	pass


###
