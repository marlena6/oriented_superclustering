Code for constrained oriented stacking of matter tracers in the cosmic web.

Inputs are file of IDs, z, ra, dec, orientation, parity x, parity y. Orientation should be defined with respect to ().
(Later on, we might include finding the orientations as part of this.)

Dependencies:
kmeans_radec: Install from https://github.com/esheldon/kmeans_radec
healpy: <pip insteall healpy --user>
pixell: <pip install pixell --user>
mpi4py: Install following instructions from https://mpi4py.readthedocs.io/en/stable/install.html
h5py: <pip install h5py>

Instructions:
(1) Open stacking_pipeline.py
(2) Update the paths and settings in the section # ALL ANALYSIS CHOICES COME BELOW #
(3) Run via <mpirun -np [# of processors] python stacking_pipeline.py>
