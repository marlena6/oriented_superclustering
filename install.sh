#!/bin/bash
set -e

# NERSC: load modules and activate your env first, before running this
# module load python
# conda activate my_stacking_environment

# MPI-enabled builds of the special packages
module load cray-hdf5-parallel
# activate stacking environment if you haven't already
MPICC="cc -shared" pip install -v --force-reinstall --no-cache-dir \
    --no-binary=mpi4py mpi4py

HDF5_MPI=ON CC=cc pip install -v --force-reinstall --no-cache-dir \
    --no-binary=h5py --no-build-isolation --no-deps h5py

# Then the project and the rest of its dependencies
pip install -e .