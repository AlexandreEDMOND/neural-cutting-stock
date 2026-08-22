"""Seeded, checkpointed and curve-persisted PyTorch training primitives.

Phase 9 introduces deep training for the quality agent. This module carries
only the reproducibility scaffolding every future trainer must reuse:

- ``set_reproducible_seed`` seeds the ``random``, NumPy and PyTorch
  generators (CUDA included) and returns the environment metadata that
  persisted artefacts must trace;
- :class:`TrainingCurves` stores validated loss/metric series under the
  versioned ``training-curves-v1`` JSON schema;
- ``save_checkpoint`` / ``load_checkpoint`` persist model weights together
  with their seed, configuration, environment and curves under the versioned
  ``neural-training-checkpoint-v1`` schema, rejecting mismatched versions at
  load time exactly like every other versioned artefact of this project.

PyTorch is an optional, versioned dependency declared by the ``learning``
extra: the classical import path never loads this module, so a missing
installation surfaces as an explicit runtime hint instead of an opaque
``ImportError``. Reproducibility is claimed for repeated seeded runs inside
the same traced environment; bit-exact equality across machines, library
builds or hardware is explicitly out of scope.
"""

import hashlib
import json
import math
import os
import platform
import random
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

TRAINING_CURVES_SCHEMA_VERSION = "training-curves-v1"
TRAINING_CHECKPOINT_SCHEMA_VERSION = "neural-training-checkpoint-v1"
_MAX_SEED = 2**32 - 1
_TORCH_INSTALL_HINT = (
    "PyTorch is required for deep training; install the versioned 'learning' extra "
    "(for example: uv sync --extra dev --extra learning)"
)


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(_TORCH_INSTALL_HINT)


def _validated_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be within [0, {_MAX_SEED}]")
    return seed


def _finite_metric(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"curve metric {name!r} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"curve metric {name!r} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class TrainingCurvePoint:
    """One validated measurement of a training run.

    ``step`` is a non-negative integer (epoch, update index...) and
    ``metrics`` maps metric names to finite floats; at least one metric is
    required so a persisted curve never contains empty measurements.
    """

    step: int
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        if not isinstance(self.metrics, Mapping) or len(self.metrics) == 0:
            raise ValueError("metrics must be a non-empty mapping")
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("metric names must be non-empty strings")
            _finite_metric(name, value)


@dataclass(frozen=True, slots=True)
class TrainingCurves:
    """Immutable series of validated training points ordered by increasing steps."""

    points: tuple[TrainingCurvePoint, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.points, tuple):
            raise ValueError("points must be a tuple")
        previous_step = -1
        for point in self.points:
            if not isinstance(point, TrainingCurvePoint):
                raise ValueError("points must contain TrainingCurvePoint instances")
            if point.step <= previous_step:
                raise ValueError("curve steps must be strictly increasing")
            previous_step = point.step

    @classmethod
    def from_points(cls, points: Iterable[TrainingCurvePoint]) -> "TrainingCurves":
        return cls(tuple(points))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": TRAINING_CURVES_SCHEMA_VERSION,
            "points": [
                {"step": point.step, "metrics": dict(point.metrics)} for point in self.points
            ],
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "TrainingCurves":
        if not isinstance(payload, Mapping):
            raise ValueError("training curves payload must be a JSON object")
        if payload.get("schema_version") != TRAINING_CURVES_SCHEMA_VERSION:
            raise ValueError(
                "unsupported training curves schema_version: "
                f"{payload.get('schema_version')!r}"
            )
        raw_points = payload.get("points")
        if not isinstance(raw_points, list):
            raise ValueError("training curves points must be a list")
        points = []
        for raw_point in raw_points:
            if not isinstance(raw_point, Mapping) or "step" not in raw_point:
                raise ValueError("each curve point must contain step and metrics")
            raw_metrics = raw_point.get("metrics")
            if not isinstance(raw_metrics, Mapping):
                raise ValueError("each curve point must contain a metrics mapping")
            points.append(TrainingCurvePoint(raw_point["step"], dict(raw_metrics)))
        return cls(tuple(points))


def set_reproducible_seed(seed: int) -> dict[str, Any]:
    """Seed Python, NumPy and PyTorch generators and return environment metadata.

    cuDNN is switched to its deterministic configuration so repeated seeded
    runs in the same traced environment reproduce identical weights. The
    returned metadata must be embedded in any artefact derived from the run.
    """

    _require_torch()
    validated = _validated_seed(seed)
    random.seed(validated)
    np.random.seed(validated)
    torch.manual_seed(validated)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(validated)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {"seed": validated, **_environment_metadata()}


def save_checkpoint(
    path: str | Path,
    *,
    module: "nn.Module",
    seed: int,
    config: Mapping[str, Any],
    curves: TrainingCurves,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Persist one versioned checkpoint and return its metadata view.

    The payload stores the module state together with the seed, the JSON
    serializable training configuration, the traced environment, the current
    curves and the checkpoint schema version. Writing is atomic: the file is
    either fully replaced or left untouched.
    """

    _require_torch()
    _validated_seed(seed)
    if not isinstance(module, nn.Module):
        raise ValueError("module must be a torch.nn.Module")
    if not isinstance(config, Mapping):
        raise ValueError("config must be a JSON object")
    canonical_config = _canonical_config(config)
    if not isinstance(curves, TrainingCurves):
        raise ValueError("curves must be a TrainingCurves instance")
    resolved_run_id = (
        run_id.strip()
        if isinstance(run_id, str) and run_id.strip()
        else _derived_run_id(seed, canonical_config)
    )
    state = {
        key: value.detach().cpu().clone() for key, value in module.state_dict().items()
    }
    payload = {
        "schema_version": TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "run_id": resolved_run_id,
        "seed": seed,
        "config": dict(config),
        "environment": _environment_metadata(),
        "curves": curves.to_payload(),
        "model_state_dict": state,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        key: value for key, value in payload.items() if key != "model_state_dict"
    }


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate one checkpoint, rejecting unsupported artefacts."""

    _require_torch()
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"checkpoint not found: {source}")
    try:
        try:
            payload = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(source, map_location="cpu")
    except Exception as error:
        raise ValueError(f"unreadable checkpoint: {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    if payload.get("schema_version") != TRAINING_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported checkpoint schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    _validated_seed(payload.get("seed"))
    if not isinstance(payload.get("config"), Mapping):
        raise ValueError("checkpoint config must be a mapping")
    if not isinstance(payload.get("environment"), Mapping):
        raise ValueError("checkpoint environment must be a mapping")
    TrainingCurves.from_payload(payload.get("curves"))
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or len(state) == 0:
        raise ValueError("checkpoint model_state_dict must be a non-empty mapping")
    return dict(payload)


def restore_module_state(module: "nn.Module", payload: Mapping[str, Any]) -> None:
    """Restore a validated checkpoint's model state into a compatible module."""

    _require_torch()
    if not isinstance(module, nn.Module):
        raise ValueError("module must be a torch.nn.Module")
    state = payload.get("model_state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping) or len(state) == 0:
        raise ValueError("payload has no model_state_dict to restore")
    try:
        module.load_state_dict(state)
    except (RuntimeError, KeyError, TypeError) as error:
        raise ValueError(f"incompatible checkpoint state: {error}") from error


def write_curves_json(path: str | Path, curves: TrainingCurves) -> None:
    """Persist training curves as stable, human-readable JSON."""

    if not isinstance(curves, TrainingCurves):
        raise ValueError("curves must be a TrainingCurves instance")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(curves.to_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_curves_json(path: str | Path) -> TrainingCurves:
    """Read back a persisted curves artefact, rejecting malformed content."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid training curves file: {path}") from error
    return TrainingCurves.from_payload(payload)


def _canonical_config(config: Mapping[str, Any]) -> str:
    try:
        return json.dumps(config, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("config must be JSON-serializable") from error


def _derived_run_id(seed: int, canonical_config: str) -> str:
    digest = hashlib.sha256(f"{seed}:{canonical_config}".encode()).hexdigest()
    return digest[:16]


def _environment_metadata() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "interpreter": sys.executable,
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__) if torch is not None else None,
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
    }


__all__ = [
    "TRAINING_CHECKPOINT_SCHEMA_VERSION",
    "TRAINING_CURVES_SCHEMA_VERSION",
    "TrainingCurvePoint",
    "TrainingCurves",
    "load_checkpoint",
    "read_curves_json",
    "restore_module_state",
    "save_checkpoint",
    "set_reproducible_seed",
    "write_curves_json",
]
