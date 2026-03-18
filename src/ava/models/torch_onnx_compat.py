"""Compatibility helpers for torch.onnx import quirks."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


def patch_torch_onnx_exporter() -> None:
	"""Ensure torch.onnx._internal.exporter resolves to exporter.py when shadowed.

	Some environments ship both `torch/onnx/_internal/exporter.py` and a
	`torch/onnx/_internal/exporter/` package, with the package taking precedence.
	This leaves `torch.onnx` unable to import DiagnosticOptions. We detect that
	shadowing and register a stub exporter module so imports succeed. The stub
	raises a clear ImportError if ONNX export features are used.
	"""
	spec = importlib.util.find_spec("torch")
	if spec is None or spec.origin is None:
		return
	torch_root = Path(spec.origin).parent
	exporter_pkg = torch_root / "onnx" / "_internal" / "exporter"
	exporter_py = exporter_pkg.parent / "exporter.py"
	if not exporter_pkg.is_dir() or not exporter_py.is_file():
		return
	init_file = exporter_pkg / "__init__.py"
	if not init_file.is_file():
		return
	try:
		init_contents = init_file.read_text(encoding="utf-8")
	except OSError:
		return
	if "DiagnosticOptions" in init_contents:
		return
	module_name = "torch.onnx._internal.exporter"
	existing = sys.modules.get(module_name)
	if existing is not None and hasattr(existing, "DiagnosticOptions"):
		return
	stub = _build_exporter_stub(module_name)
	sys.modules[module_name] = stub


def _build_exporter_stub(module_name: str) -> types.ModuleType:
	stub = types.ModuleType(module_name)
	message = (
		"torch.onnx exporter is unavailable because the torch installation "
		"exposes both torch/onnx/_internal/exporter.py and "
		"torch/onnx/_internal/exporter/. Reinstall torch to remove the "
		"shadowing package."
	)

	class _OnnxExporterUnavailable(ImportError):
		pass

	def _raise_unavailable(*_args, **_kwargs):
		raise _OnnxExporterUnavailable(message)

	class DiagnosticOptions:
		def __init__(self, *args, **kwargs):
			_raise_unavailable(*args, **kwargs)

	class ExportOptions:
		def __init__(self, *args, **kwargs):
			_raise_unavailable(*args, **kwargs)

	class ONNXProgram:
		def __init__(self, *args, **kwargs):
			_raise_unavailable(*args, **kwargs)

	class ONNXProgramSerializer:
		def __init__(self, *args, **kwargs):
			_raise_unavailable(*args, **kwargs)

	class ONNXRuntimeOptions:
		def __init__(self, *args, **kwargs):
			_raise_unavailable(*args, **kwargs)

	class InvalidExportOptionsError(RuntimeError):
		pass

	class OnnxExporterError(RuntimeError):
		pass

	class OnnxRegistry:
		def __init__(self, *args, **kwargs):
			_raise_unavailable(*args, **kwargs)

	def dynamo_export(*args, **kwargs):
		_raise_unavailable(*args, **kwargs)

	def enable_fake_mode(*args, **kwargs):
		_raise_unavailable(*args, **kwargs)

	stub.DiagnosticOptions = DiagnosticOptions
	stub.ExportOptions = ExportOptions
	stub.ONNXProgram = ONNXProgram
	stub.ONNXProgramSerializer = ONNXProgramSerializer
	stub.ONNXRuntimeOptions = ONNXRuntimeOptions
	stub.InvalidExportOptionsError = InvalidExportOptionsError
	stub.OnnxExporterError = OnnxExporterError
	stub.OnnxRegistry = OnnxRegistry
	stub.dynamo_export = dynamo_export
	stub.enable_fake_mode = enable_fake_mode
	return stub
