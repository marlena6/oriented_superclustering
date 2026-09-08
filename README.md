Code for constrained oriented stacking of matter tracers in the cosmic web.

Dependencies:

(If installing the following in a conda environment, use 'conda install' instead of pip install)

kmeans_radec: Install from https://github.com/esheldon/kmeans_radec

healpy: ```pip install healpy --user```

pixell: ```pip install pixell --user```

mpi4py: Install following instructions from https://mpi4py.readthedocs.io/en/stable/install.html. For NERSC users, mpi4py installation or loading must follow NERSC-specific instructions.

h5py: ```pip install h5py --user```

Installation:
Install with an editable installation via:
```pip install -e .```

Instructions:

(1) Copy the example config files from examples/ to a new directory.

(2) If doing oriented stacking, input the desired inputs to the config_orient.yaml. Each line is commented with instructions.

(3) If doing oriented stacking, run the orientations. Ideally, use 1 task and many (e.g. 24-48) processes. 
```python scripts/orient_pipeline.py path/to/config_orient.yaml```

Outputs will be in format <[galaxy sample for stack centers]_[cuts on that sample]_[galaxies used for constraints and orientation]_[constraints placed using the smooth galaxy field, including smoothing scale]_[orientation scale]_[percentage of galaxy data used for constraints and orientation]

(4) Stacking: Input the desired inputs to your config_orient.yaml. If you want to run in 'test' mode, set test=True and you will run a smaller amount of objects (setting nObj to 1000, for example, should run in 1.7 minutes, 10,000 in 17 minutes, but this depends on number of cores)

(5) Ideally, run with use_mpi=True on a node with ntasks = number of regions for subsampling, e.g. 48. Depending on the system, run with:

 ```mpirun -np [# of processors] python stacking_pipeline.py``` or ```srun --cpu-bind=cores python scripts/stacking_pipeline.py path/to/config_stack.yaml```
