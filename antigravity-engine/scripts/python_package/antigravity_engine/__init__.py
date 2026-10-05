"""
Antigravity Engine Python SDK
"""
import os
import ctypes
from pathlib import Path
from .native_bridge import NativeMetalEngine, AntigravityConfig

__version__ = "2.5.0"

def get_dylib_path() -> str:
    pkg_dir = Path(__file__).parent.resolve()
    dylib = pkg_dir / "lib" / "libantigravity_engine.dylib"
    if dylib.exists():
        return str(dylib)
    dylib_direct = pkg_dir / "libantigravity_engine.dylib"
    if dylib_direct.exists():
        return str(dylib_direct)
    # Check current directory / parent directories
    for candidate in [
        Path.cwd() / "libantigravity_engine.dylib",
        Path(__file__).resolve().parent.parent / "libantigravity_engine.dylib",
        # ANTIGRAVITY_DYLIB_DIR replaces the absolute paths of one developer's machine
        # that used to be hardcoded here, and which no installing user could ever have.
        *(
            [Path(os.environ["ANTIGRAVITY_DYLIB_DIR"]) / "libantigravity_engine.dylib"]
            if os.environ.get("ANTIGRAVITY_DYLIB_DIR") else []
        ),
    ]:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"libantigravity_engine.dylib not found in package {pkg_dir}")

def create_engine(
    model_path: str = None,
    n_channels: int = 8,
    vocab_size: int = 32000,
    hidden_dim: int = 2048,
    max_seq_len: int = 2048
) -> NativeMetalEngine:
    return NativeMetalEngine(
        dylib_path=get_dylib_path(),
        model_path=model_path,
        n_channels=n_channels,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len
    )

__all__ = ["NativeMetalEngine", "AntigravityConfig", "create_engine", "get_dylib_path", "__version__"]
