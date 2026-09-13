from setuptools import setup, find_packages
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        # Package contains compiled ARM64 dylib for macOS
        return "py3", "none", "macosx_12_0_arm64"

setup(
    name="antigravity-engine",
    version="2.5.0",
    description="High-Performance Edge AI Inference Engine for Apple Silicon",
    packages=["antigravity_engine"],
    package_data={
        "antigravity_engine": [
            "lib/*.dylib",
            "*.dylib",
            "shaders/*.metallib"
        ]
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22.0",
        "cryptography>=41.0.0"
    ],
    cmdclass={"bdist_wheel": bdist_wheel}
)
