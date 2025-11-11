program_path = "/home/mlokken/oriented_stacking/oriented_superclustering/"

# import stuff from the parent directory (could update to be proper package later)
import sys
sys.path.append(program_path)
import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from pixell import enmap
import catalog
from kmeans_radec import kmeans_sample
from stacking_functions import Chunk, stackChunk
import h5py
from pathlib import Path
import os
import shutil

# restart run from previous try?
restart_run = False# make a new directory for all the inputs and outputs of this run, for bookkeeping
newdir_name = "flhiagn_ex1000fast/"
    
use_mpi = True
if use_mpi:
    from mpi4py import MPI

    # get the MPI ingredients
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
else:
    rank = 0
    size = 1
# ALL ANALYSIS CHOICES COME BELOW
########################################################

errors = True  # if true, split regions to get error estimates
if errors:
    nreg = 3  # number of chunks = number of processors to use
else:
    nreg = 1
    
# Describe the constraints on the input catalog
constraint_str = "lrg11p2_elg2e-4+lrg11p2_nu10gt2_e10gtpt3_o10_100pct_0.95_1.05"

orient = "asym_xy"  # options are "original", "random", "sym", "asym_x", "asym_y", "asym_xy"

cutout_rad = 20.0 * u.Mpc  # size of the cutout in comoving Mpc

dz_rescale = 0.01  # size of z bins for rescaling

basepath = "/mnt/scratch-lustre/mlokken/stacking/flamingo/orient_by_flam_elg2e-4+lrg_11p2_100/stacks/enmap/"

stack_pts_path = "/mnt/raid-cita/mlokken/data/flamingo/stacking_points/"

stack_pts_file = f"{constraint_str}.csv"

test = True # if 'test', a smaller amount of objects will be run
########################################################

if test:
    nObj = 1000
    teststr = f'_test{nObj:.1e}'
else:
    nObj = None
    teststr = ''
    
savepath = basepath + newdir_name + "/"
# have rank 0 make the new directory, all others wait
if rank == 0:
    assert not os.path.exists(newdir_name) or restart_run, f"Directory {newdir_name} already exists. If you want to restart the run, set restart_run=True."
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        print(f"Created directory {savepath} for this run.")
    if restart_run:
        print(f"Restarting run in existing directory {savepath}.")
if use_mpi and size > 1:
    comm.Barrier()  # wait for rank 0 to finish making the directory
    
# input the dictionary of maps to stack here
# format should be "type":"path/to/map.fits"
# maps should be enmaps
# eventually this can include the masks as well
maps = {
    "map1": {
        "type": "y",
        "path": "/mnt/raid-cita/mlokken/data/flamingo/ComptonY_strongest_agn_enmap_res1a.fits",
        "shortname": "fl_hiagn",
    }
}

# Make sure the output file doesn't already exist
stkpts_str = Path(stack_pts_file).stem
if len(maps) == 1:
    outfile = f"{savepath}/{maps['map1']['shortname']}_consol_stacks_{stkpts_str}{teststr}.h5"
else:    
    print("Not yet implemented.")
    
print("Will save to", outfile)
assert not os.path.exists(outfile), f"Final output file already exists at: {outfile}"


# add some function here to check if each map is an enmap, and convert from healpix to enmap otherwise

# read the orientation information

orientfile = stack_pts_path + stack_pts_file

if rank == 0:
    # save a copy of the orient file in the new directory for bookkeeping
    shutil.copy(orientfile, savepath + stack_pts_file)
    # save a copy of this script in the new directory for bookkeeping
    shutil.copy(__file__, savepath + Path(__file__).name)
    
Cat = catalog.Catalog(
    name="standard",
    nameLong=constraint_str,
    pathInCatalog=stack_pts_path + stack_pts_file,
    nObj=nObj
)
print("Analyzing catalog of length", len(Cat.Z))
minz = np.amin(Cat.Z)
maxz = np.amax(Cat.Z)
print(f"Redshift range in catalog: {minz:.3f} - {maxz:.3f}")
#### getting the region splits for errors ####
if rank == 0:
    km = kmeans_sample(np.vstack((Cat.RA, Cat.DEC)).T, nreg, maxiter=100, tol=1.0e-5)
    labels = km.labels
    if size > 1:
        for i in range(1, size):
            comm.Send(labels, dest=i)
            print(f"sending labels to rank {i}")
elif rank > 0:
    labels = np.empty(len(Cat.RA), dtype=np.int64)
    comm.Recv(labels, source=0)
    print(f"received labels on rank {rank}")
Cat.labels = labels  # add labels to the Catalog object


# if the unit of cutout_rad is Mpc, then we need to convert it to degrees
if cutout_rad.unit == u.Mpc:
    # based on the minimum redshift, set the max size for the cutouts
    cutout_rad_deg = (
        1 / (cosmo.kpc_comoving_per_arcmin(minz).to(u.Mpc / u.deg)) * cutout_rad
    )
elif cutout_rad.unit in [u.arcmin, u.arcsec]:
    cutout_rad_deg = cutout_rad.to(u.deg)
else:
    raise ValueError(
        "cutout_rad must be in units of Mpc, degrees, arcminutes, or arcseconds"
    )

cutout_resolution = (0.5 * u.arcmin).to(u.deg)
print(
    f"will take thumbnails with size {cutout_rad_deg:.2f} and resolution {cutout_resolution:.2f}."
)

# calculate the comoving size and physical size of the lowest-redshift cutout
Mpc_per_deg_comov_zmin = cosmo.kpc_comoving_per_arcmin(minz).to(u.Mpc / u.degree)
Mpc_per_deg_phys_zmin  = cosmo.kpc_proper_per_arcmin(minz).to(u.Mpc / u.degree)

### setup multiprocessing ###
if use_mpi:
    nruns_local = nreg // size
    if rank == size - 1:
        extras = nreg % size
    else:
        extras = 0
else:
    nruns_local = nreg
    extras = 0
### end setup multiprocessing ###


# Prepare to save to an HDF5 file
file_i = f"{savepath}/stacks_{stkpts_str}_{rank}{teststr}.h5"
if os.path.exists(file_i):
    assert restart_run, f"File {file_i} already exists. If you want to restart the run, set restart_run=True."

with h5py.File(f"{savepath}/stacks_{stkpts_str}_{rank}{teststr}.h5", "w") as f:
    f.attrs["cutout_rad_deg"]   = cutout_rad_deg.value
    f.attrs["cutout_rad_cMpc"] = cutout_rad_deg * Mpc_per_deg_comov_zmin.value
    f.attrs["cutout_rad_pMpc"]   = cutout_rad_deg * Mpc_per_deg_phys_zmin.value
    for m in maps:
        mappath = maps[m]["path"]
        sn = maps[m]["shortname"]
        map_group = f.create_group(sn)
        map_group.attrs['map_path'] = mappath
        print(f"Reading map {mappath}")
        # imap = enmap.read_map(maps[m]["path"])
        for i in range(nruns_local + extras):
            n = rank * nruns_local + i
            in_reg = Cat.labels == n
            nobj_regn = 0
            # make an HDF5 group for this region
            reg_group = map_group.create_group(f"reg_{n}")
            reg_group.attrs["Region"] = n
            print(f"Analyzing region {n}")
            for z in np.linspace(
                minz, maxz, int((maxz - minz) / dz_rescale)
            ):  # iterate through small z bins
                Mpc_per_deg_phys_z  = cosmo.kpc_proper_per_arcmin(z).to(u.Mpc / u.degree)
                Mpc_per_deg_comov_z = cosmo.kpc_comoving_per_arcmin(z).to(u.Mpc / u.degree)
                phys_rescale_factor = Mpc_per_deg_phys_zmin / Mpc_per_deg_phys_z
                comov_rescale_factor= Mpc_per_deg_comov_zmin / Mpc_per_deg_comov_z
                inz = (Cat.Z < (z + dz_rescale)) & (Cat.Z > (z - dz_rescale))
                z_rescale_str = f"z_{(z - dz_rescale):.2f}_{(z + dz_rescale):.2f}"
                z_group = reg_group.create_group(z_rescale_str)  # create a subgroup
                # make the ChunkObj for these z
                ChunkObj = Chunk(
                    Cat.RA[in_reg & inz],
                    Cat.DEC[in_reg & inz],
                    Cat.alpha[in_reg & inz],
                    Cat.x_asym[in_reg & inz],
                    Cat.y_asym[in_reg & inz],
                )
                # define the map edges for this region
                lowra, highra = np.amin(Cat.RA[in_reg])-cutout_rad_deg.value, np.amax(Cat.RA[in_reg])+cutout_rad_deg.value
                lowdec, highdec = np.amin(Cat.DEC[in_reg])-cutout_rad_deg.value, np.amax(Cat.DEC[in_reg])+cutout_rad_deg.value
                imap = enmap.read_map(maps[m]["path"], box=[[np.radians(lowdec),np.radians(lowra)],[np.radians(highdec),np.radians(highra)]])
                # get the stack
                print("Stacking region", n, "at z", z)
                stack_n, stack_n_phys, stack_n_comov = stackChunk(
                    ChunkObj,
                    imap,
                    cutout_rad_deg.value,
                    cutout_resolution.value,
                    orient=orient,
                    rescale_1 = phys_rescale_factor,
                    rescale_2 = comov_rescale_factor,
                )
                # save to this delta-z subgroup
                z_group.attrs["Nobj"]=ChunkObj.nObj
                z_group.create_dataset("stack_deg", data=stack_n)
                z_group.create_dataset("stack_phys", data=stack_n_phys)
                z_group.create_dataset("stack_comov", data=stack_n_comov)
                z_group.create_dataset("RA", data=Cat.RA[in_reg & inz])
                z_group.create_dataset("dec", data=Cat.DEC[in_reg & inz])
                z_group.create_dataset("z", data=Cat.Z[in_reg & inz])
                nobj_regn += ChunkObj.nObj
            reg_group.attrs["Nobj"] = nobj_regn

if use_mpi and size > 1:
    # wait for the others to finish writing
    comm.Barrier()
    # collect all
    if rank == 0 and size > 1:
        import glob

        with h5py.File(outfile, "w") as consol_f:
            files = glob.glob(f"{savepath}/stacks_{stkpts_str}*{teststr}.h5")
            for m in maps:
                mappath = maps[m]["path"]
                sn = maps[m]["shortname"]
                consol_f.create_group(sn)
                for fname in sorted(files):
                    with h5py.File(fname, "r") as mpif:
                        if f'_0{teststr}.h5' in fname: # if the rank_0 file, copy over the attributes (only need to do once)
                            consol_f.attrs["cutout_rad_deg"] = mpif.attrs["cutout_rad_deg"]
                            consol_f.attrs["cutout_rad_cMpc"] = mpif.attrs["cutout_rad_cMpc"]
                            consol_f.attrs["cutout_rad_pMpc"] = mpif.attrs["cutout_rad_pMpc"]
                            consol_f[sn].attrs['map_path'] = mpif[sn].attrs['map_path']
                        for group in mpif[sn].keys():
                            mpif[sn].copy(mpif[sn][group], consol_f[sn], name=group)
            for file in files:
                print(f"Removing {file}")
                os.remove(file)
        print(f"Saved to {outfile}")
else:
    os.rename(f"{savepath}/stacks_{stkpts_str}_{rank}{teststr}.h5", outfile)
    print(f"Saved to {outfile}")

