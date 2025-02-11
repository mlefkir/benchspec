# benchspec

This repository contains benchmarks for X-ray spectral fitting packages. It is intended to be used for testing and comparing the performance of different packages.

## Todo

- [ ] Add jaxspec
- [ ] Add benchmarks for models evaluation, gradients
- [ ] Comparison with other packages: xspec, elisa, SpectralFitting.jl...

## Installation

Create a conda environment with the required packages:
```bash
conda create -n benchspec python=3.12
conda activate benchspec
```

Install the poetry package and clone the repository:
```bash
pip install poetry
git clone https://github.com/mlefkir/benchspec.git
```

Install the required packages:
```bash
cd benchspec
poetry install
```
