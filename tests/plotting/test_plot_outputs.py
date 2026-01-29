import numpy as np
import pytest

from ava.plotting.grid_plot import grid_plot
from ava.plotting.mmd_plots import mmd_matrix_plot


def test_grid_plot_writes_file(tmp_path):
	rng = np.random.default_rng(0)
	specs = rng.random((2, 2, 8, 8))
	filename = tmp_path / "plots" / "grid.pdf"
	grid_plot(specs, filename=str(filename))
	assert filename.exists()
	assert filename.stat().st_size > 0


def test_projection_plot_writes_file(tmp_path):
	pytest.importorskip("umap")
	from ava.plotting.latent_projection import projection_plot

	rng = np.random.default_rng(1)
	embed = rng.normal(size=(12, 2))
	colors = np.linspace(0.0, 1.0, len(embed))
	filename = tmp_path / "plots" / "latent.png"
	projection_plot(embed, color=colors, save_filename=str(filename))
	assert filename.exists()
	assert filename.stat().st_size > 0


def test_mmd_matrix_plot_writes_file(tmp_path):
	mmd2 = np.array([[0.0, 1.0], [1.0, 0.0]])
	filename = tmp_path / "plots" / "mmd_matrix.pdf"
	mmd_matrix_plot(mmd2, filename=str(filename))
	assert filename.exists()
	assert filename.stat().st_size > 0


def test_tooltip_plot_writes_assets(tmp_path, monkeypatch):
	pytest.importorskip("bokeh")
	from ava.plotting import tooltip_plot as tooltip_module
	from bokeh.io import save

	monkeypatch.setattr(tooltip_module, "show", lambda fig: save(fig))
	rng = np.random.default_rng(2)
	embed = rng.normal(size=(12, 2))
	images = rng.random((12, 8, 8))
	output_dir = tmp_path / "tooltip"

	tooltip_module.tooltip_plot(
		embed,
		images,
		output_dir=str(output_dir),
		num_imgs=3,
		n=10,
	)

	html_file = output_dir / "main.html"
	img_file = output_dir / "0.jpg"
	assert html_file.exists()
	assert html_file.stat().st_size > 0
	assert img_file.exists()
	assert img_file.stat().st_size > 0
