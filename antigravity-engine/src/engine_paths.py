"""
Locating the files the native engine needs.

Shared by the benchmark harnesses so they cannot disagree about what a model
directory contains — the accuracy harness originally passed the directory itself
where a .safetensors file was required, and the throughput harness hardcoded
"model.safetensors", so an oddly named checkpoint failed in one and a sharded one
would have been half-loaded by both.
"""

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def resolve_weights(model_dir: PathLike) -> Path:
    """Find the safetensors file inside a model directory.

    load_weights() takes a .safetensors FILE: AntigravityEngineLoadModel opens the
    path directly and parseSafetensors reads its header from it. A directory fails
    at the open().

    Raises FileNotFoundError when there is nothing to load, or when the checkpoint
    is sharded — the engine's parser reads a single file, so picking one shard
    would load part of a model and generate from it without any sign of trouble.
    """
    directory = Path(model_dir)
    candidates = sorted(directory.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"no .safetensors file in {directory}")

    for preferred in ("model.safetensors", "pytorch_model.safetensors"):
        for candidate in candidates:
            if candidate.name == preferred:
                return candidate

    if len(candidates) > 1:
        raise FileNotFoundError(
            f"{directory} holds {len(candidates)} safetensors files "
            f"({', '.join(c.name for c in candidates)}); the engine's parser reads "
            f"one file, so pass a merged checkpoint or name it model.safetensors")
    return candidates[0]


def find_dylib(explicit: Optional[PathLike] = None) -> Optional[Path]:
    """The engine dylib, if it can be found without the Python package installed.

    NativeMetalEngine searches the installed-package locations. A build straight
    out of the repository is not in any of them, which is how CI and a local build
    both reach it.
    """
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None

    repo = Path(__file__).resolve().parent.parent
    for candidate in (
        repo / "build" / "lib" / "libantigravity_engine.dylib",
        repo / "scripts" / "python_package" / "antigravity_engine" / "lib"
            / "libantigravity_engine.dylib",
        repo / "libantigravity_engine.dylib",
    ):
        if candidate.is_file():
            return candidate
    return None
