from setuptools import setup, find_packages
from pathlib import Path
import os

if __name__ == "__main__":
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    def _read_reqs(relpath):
        fullpath = os.path.join(os.path.dirname(__file__), relpath)
        with open(fullpath) as f:
            return [
                s.strip()
                for s in f.readlines()
                if (s.strip() and not s.startswith("#"))
            ]

    REQUIREMENTS = _read_reqs("requirements.txt")

    setup(
        name="dream_bench",
        packages=find_packages(),
        include_package_data=True,
        version="0.0.1",
        license="MIT",
        description="A tool for benchmarking image generation models.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Zion English",
        author_email="zion.m.english@gmail.com",
        url="https://github.com/nousr/dream-bench",
        data_files=[(".", ["README.md"])],
        keywords=["machine learning", "txt2im", "benchmark", "pytorch"],
        install_requires=REQUIREMENTS,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.10",
        ],
    )
