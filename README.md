Code for constrained oriented stacking of matter tracers in the cosmic web.

Inputs are file of IDs, z, ra, dec, orientation, parity x, parity y. Orientation should be defined with respect to ().
(Later on, we might include finding the orientations as part of this.)

Dependencies:

(If installing the following in a conda environment, use 'conda install' instead of pip install)

kmeans_radec: Install from https://github.com/esheldon/kmeans_radec

healpy: '''pip install healpy --user'''

pixell: '''pip install pixell --user'''

mpi4py: Install following instructions from https://mpi4py.readthedocs.io/en/stable/install.html

h5py: '''pip install h5py --user'''

Instructions:

(1) Open stacking_pipeline.py

(2) Get the fiducial y map by: '''curl -s --retry 5 -k -o ilc_actplanck_ymap.fits https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_adv/Compton_y_maps/ilc_actplanck_ymap.fits''' (1.78 GB)

(3) Get the file with orientation information from '''/global/cfs/projectdirs/desi/users/mlokken/oriented_stacks/<orientation_directory>''', each configuration file will be in a directory that describes the galaxies used for orientation '''orient_by_<sample>_<percentage>''' and be titled '''<constraints>.csv''' where the constraints are formatted as <[galaxy sample for stack centers]_[cuts on that sample]_[galaxies used for constraints and orientation]_[constraints placed using the smooth galaxy field, including smoothing scale]_[orientation scale]_[percentage of galaxy data used for constraints and orientation]

(4) Update the paths and settings in stacking_pipeline.py, in the section following '''# ALL ANALYSIS CHOICES COME BELOW #'''. If you want to run in 'test' mode, set test=True and you will run a smaller amount of objects (setting nObj to 1000, for example, should run in 1.7 minutes, 10,000 in 17 minutes)

(5) Run via '''mpirun -np [# of processors] python stacking_pipeline.py'''