import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from pixell import enmap
import catalog
from kmeans_radec import kmeans_sample
from stacking_functions import Chunk, stackChunk, rescale_prof
from stack_statistics import radial_decompose_2D
import h5py
from pathlib import Path
import os



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
    nreg = 48  # number of chunks = number of processors to use
else:
    nreg = 1
    
# Describe the constraints on the input catalog
constraint_str = "lrgs_zlim_elgclrgc_nu10gt2_e10gtpt3_o10_100pct_0.81_0.94"

orient = "asym_xy"  # options are "original", "random", "sym", "asym_x", "asym_y", "asym_xy"

cutout_rad = 20.0 * u.Mpc  # size of the cutout in comoving Mpc

dz_rescale = 0.01  # size of z bins for rescaling

savepath = "/mnt/scratch-lustre/mlokken/stacking/ACTxDESI/orient_by_desi_elgc+lrgc_100/stacks/enmap/"

stack_pts_path = "/mnt/raid-cita/mlokken/data/desi/stacking_points/"

stack_pts_file = f"{constraint_str}.csv"

test = False # if 'test', a smaller amount of objects will be run
########################################################

if test:
    nObj = 10000
    teststr = '_test'
else:
    nObj = None
    teststr = ''
    
# Make sure the output file doesn't already exist
stkpts_str = Path(stack_pts_file).stem
outfile = f"{savepath}/consol_stacks_{stkpts_str}{teststr}.h5"
assert not os.path.exists(outfile), f"Final output file already exists at: {outfile}"


# add some function here to check if each map is an enmap, and convert from healpix to enmap otherwise

# read the orientation information


Cat = catalog.Catalog(
    name="standard",
    nameLong=constraint_str,
    pathInCatalog=stack_pts_path + stack_pts_file,
    nObj=nObj
)
print("Analyzing catalog of length", len(Cat.Z))
minz = np.amin(Cat.Z)
maxz = np.amax(Cat.Z)

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
#### getting the region splits for errors ####

# input the dictionary of maps to stack here
# format should be "type":"path/to/map.fits"
# maps should be enmaps
# eventually this can include the masks as well
maps = {
    "map1": {
        "type": "y",
        "path": "/mnt/raid-cita/mlokken/data/act_ymaps/ilc_SZ_yy.fits",
        "shortname": "ACT_y_fid",
    }
}

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
        imap = enmap.read_map(maps[m]["path"])
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
                # find the physical and comoving rescaling factors
                
                # get the stack
                stack_n, stack_n_phys, stack_n_comov = stackChunk(
                    ChunkObj,
                    imap,
                    cutout_rad_deg.value,
                    cutout_resolution.value,
                    orient=orient,
                    rescale_1 = phys_rescale_factor,
                    rescale_2 = comov_rescale_factor,
                )
                # get the profiles
                r_deg, Cr_ang, Sr_ang = radial_decompose_2D(stack_n, 5, cutout_rad_deg.value)
                r_pMpc = r_deg * Mpc_per_deg_phys_z.value
                r_cMpc = r_deg * Mpc_per_deg_comov_z.value
                if z==minz:
                    # save the 'base' r coordinates in comoving and physical size
                    base_r_cMpc = r_cMpc
                    base_r_pMpc = r_pMpc
                    Cr_comov = Cr_phys = Cr_ang  # the profiles do not need to be cropped and rescaled
                    Sr_comov = Sr_phys = Sr_ang
                else:
                    Cr_comov = rescale_prof(Cr_ang, r_cMpc, base_r_cMpc)
                    Cr_phys  = rescale_prof(Cr_ang, r_pMpc, base_r_pMpc)
                    Sr_comov = rescale_prof(Sr_ang, r_cMpc, base_r_cMpc)
                    Sr_phys  = rescale_prof(Sr_ang, r_pMpc, base_r_pMpc)
                # save to this delta-z subgroup
                z_group.attrs["Nobj"]=ChunkObj.nObj
                z_group.create_dataset("Cr_deg_profiles", data=Cr_ang)
                z_group.create_dataset("Sr_deg_profiles", data=Sr_ang)
                z_group.create_dataset("Cr_prop_profiles", data=Cr_phys)
                z_group.create_dataset("Sr_prop_profiles", data=Sr_phys)
                z_group.create_dataset("Cr_comov_profiles", data=Cr_comov)
                z_group.create_dataset("Sr_comov_profiles", data=Sr_comov)
                z_group.create_dataset("r_deg", data=r_deg)
                z_group.create_dataset("r_prop_Mpc", data=r_pMpc)
                z_group.create_dataset("r_comov_Mpc", data=r_cMpc)
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

