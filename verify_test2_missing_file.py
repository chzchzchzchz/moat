import sys
sys.path.insert(0, 'antigravity-engine/src')

from native_bridge import NativeMetalEngine

print("Testing loading from non-existent safetensors path: 'models/tinyllama/NON_EXISTENT_FILE.safetensors'")
try:
    engine = NativeMetalEngine(
        dylib_path='libantigravity_engine.dylib',
        model_path='models/tinyllama/NON_EXISTENT_FILE.safetensors',
        n_channels=8
    )
    print("ERROR: Engine returned successfully despite missing file! (FAIL)")
except Exception as e:
    print(f"SUCCESS: Engine failed to load as expected with error: {e}")
