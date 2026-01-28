import sys
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
import yaml
import glob
import filecmp
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord


# Load config
if len(sys.argv) != 2:
    raise ValueError("Please provide a config yaml file as an argument.")
config_file_path = sys.argv[1]
print(f"Loading config from {config_file_path}")
with open(config_file_path, "r") as f:
    cfg = yaml.safe_load(f)
print(cfg["mpi"])
use_mpi = cfg["mpi"]["use_mpi"]
if use_mpi:
    from mpi4py import MPI

    # get the MPI ingredients
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
else:
    rank = 0
    size = 1

restart_run = cfg["run"]["restart_run"]
newdir_name = cfg["run"]["newdir_name"]
test = cfg["run"]["test"]
if test:
    nObj = cfg["run"]["nObj_test"]
    teststr = f"_test{nObj:.1e}"
else:
    nObj = None
    teststr = ""
errors = cfg["errors"]["enabled"]

if errors:
    nreg = cfg["errors"]["nreg"]
else:
    nreg = 1

constraint_str = cfg["catalog"]["constraint_str"]
orient = cfg["analysis"]["orient"]
cutout_rad = cfg["analysis"]["cutout_rad_mpc"] * u.Mpc
dz_rescale = cfg["analysis"]["dz_rescale"]
zmin = cfg["analysis"]["zmin"]
zmax = cfg["analysis"]["zmax"]
basepath = cfg["paths"]["basepath"]
stack_pts_path = cfg["paths"]["stack_pts_path"]
stack_pts_file = f"{constraint_str}" if f"{constraint_str}".endswith(".csv") else f"{constraint_str}.csv"
savepath = os.path.join(basepath, newdir_name)

# have rank 0 make the new directory, all others wait
if rank == 0:
    assert not os.path.exists(newdir_name) or restart_run, (
        f"Directory {newdir_name} already exists. If you want to restart the run, set restart_run=True."
    )
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        print(f"Created directory {savepath} for this run.")
    if restart_run:
        print(f"Restarting run in existing directory {savepath}.")
if use_mpi and size > 1:
    comm.Barrier()  # wait for rank 0 to finish making the directory

maps = cfg["maps"]
stkpts_str = Path(stack_pts_file).stem
if len(maps) == 1:
    outfile = (
        f"{savepath}/{maps['map1']['shortname']}_consol_stacks_z{zmin:.2f}_{zmax:.2f}_{stkpts_str}{teststr}.h5"
    )
else:
    print("Not yet implemented.")

# Make sure the output file doesn't already exist
if rank == 0:
    print("Will save to", outfile)
assert not os.path.exists(outfile), f"Final output file already exists at: {outfile}"

# add some function here to check if each map is an enmap, and convert from healpix to enmap otherwise

# read the orientation information

orientfile = stack_pts_path + stack_pts_file

if rank == 0:
    # if not already there, save a copy of the orient file in the new directory for bookkeeping
    if not os.path.exists(savepath + stack_pts_file):
        shutil.copy(orientfile, savepath + stack_pts_file)
    # if no yaml file is in the new directory yet, save a copy of the config file for bookkeeping
    yamls = glob.glob(savepath + "/*.yaml")
    assert len(yamls) <= 1, (
        f"Multiple yaml files found in {savepath}. Please ensure only one config file is present."
    )
    if len(yamls) == 1:
        assert filecmp.cmp(yamls[0], config_file_path, shallow=True), (
            f"YAML file {yamls[0]} does not match the config file used for this run: {config_file_path}. Delete the old yaml file or set a different newdir_name."
        )
    if yamls == []:
        shutil.copy(config_file_path, savepath + "/config_used.yaml")

if rank==0:
    # read the catalog
    Cat = catalog.Catalog(
        name="standard",
        nameLong=constraint_str,
        pathInCatalog=stack_pts_path + stack_pts_file,
        nObj=nObj,
    )
    print("Analyzing catalog of length", len(Cat.Z))
    if size > 1:
            for i in range(1, size):
                comm.send(Cat, dest=i)
                print(f"sending catalog to rank {i}")
elif rank > 0:
    Cat = comm.recv(source=0)
    print(f"received catalog on rank {rank} of length", len(Cat.Z))
    
if zmin is None:
    zmin = np.amin(Cat.Z)
if zmax is None:
    zmax = np.amax(Cat.Z)
print(f"Redshift range to stack: {zmin:.3f} - {zmax:.3f}")
#### getting the region splits for errors ####
# if labels file already exists, read from that
labels_file = f"{savepath}/region_labels_{nreg}reg{teststr}.txt"
if restart_run:
    assert os.path.exists(labels_file), (
        f"Region labels file {labels_file} not found. Cannot restart run without it. Set restart_run=False to generate new region labels."
    )
if os.path.exists(labels_file) and restart_run:
    labels = np.loadtxt(labels_file, dtype=np.int64)
    print(f"Read region labels from {labels_file}")
else:
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
    if rank == 0:
        np.savetxt(labels_file, labels, fmt="%d")
        print(f"Saved region labels to {labels_file}")
        colors = ['C'+str(i) for i in range(10)]
        rands  = np.random.choice(np.arange(len(Cat.RA)), size=1000, replace=False)
        plt.scatter(Cat.RA[rands], Cat.DEC[rands], c=[colors[l%10] for l in labels[rands]], s=5)
        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=f'Region {i}', markerfacecolor=colors[i%10], markersize=5) for i in range(nreg)],loc='best')
        plt.savefig(f"{savepath}/region_splits.png")
        plt.clf()
Cat.labels = labels  # add labels to the Catalog object
print(np.unique(labels), "labels")

    

# if the unit of cutout_rad is Mpc, then we need to convert it to degrees
if cutout_rad.unit == u.Mpc:
    # based on the minimum redshift, set the max size for the cutouts
    cutout_rad_deg = (
        1 / (cosmo.kpc_comoving_per_arcmin(zmin).to(u.Mpc / u.deg)) * cutout_rad
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
Mpc_per_deg_comov_zmin = cosmo.kpc_comoving_per_arcmin(zmin).to(u.Mpc / u.degree)
Mpc_per_deg_phys_zmin = cosmo.kpc_proper_per_arcmin(zmin).to(u.Mpc / u.degree)

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
if not os.path.exists(file_i):
    with h5py.File(f"{savepath}/stacks_{stkpts_str}_{rank}{teststr}.h5", "w") as f:
        f.attrs["cutout_rad_deg"] = cutout_rad_deg.value
        f.attrs["cutout_rad_cMpc"] = cutout_rad_deg * Mpc_per_deg_comov_zmin.value
        f.attrs["cutout_rad_pMpc"] = cutout_rad_deg * Mpc_per_deg_phys_zmin.value
        for m in maps:
            mappath = maps[m]["path"]
            sn = maps[m]["shortname"]
            map_group = f.create_group(sn)
            map_group.attrs["map_path"] = mappath
            print(f"Reading map {mappath}")
            if size==1:
                # read the whole map
                imap = enmap.read_map(maps[m]["path"])
            for i in range(nruns_local + extras):
                n = rank * nruns_local + i
                in_reg = Cat.labels == n
                print(f"Rank {rank}, region {n}, Nobj = {in_reg.sum()}")

                nobj_regn = 0
                # make an HDF5 group for this region
                reg_group = map_group.create_group(f"reg_{n}")
                reg_group.attrs["Region"] = n
                print(f"Analyzing region {n}")
                # define the map edges for this region
                sc = SkyCoord(ra=Cat.RA[in_reg]*u.deg, dec=Cat.DEC[in_reg]*u.deg, frame="icrs")
                ra_wrapped = sc.ra.wrap_at(180*u.deg)
                lowra, highra = (
                    ra_wrapped.min() - (cutout_rad_deg+0.5*u.deg),
                    ra_wrapped.max() + (cutout_rad_deg+0.5*u.deg),
                )
                lowdec, highdec = (
                    sc.dec.min() - (cutout_rad_deg+0.5*u.deg),
                    sc.dec.max() + (cutout_rad_deg+0.5*u.deg),
                )
                
                # plot a subset of this region and save
                # rands = np.random.choice(np.arange(len(Cat.RA[in_reg])), size=min(1000,len(Cat.RA[in_reg])), replace=False)
                # plt.scatter(Cat.RA[in_reg][rands], Cat.DEC[in_reg][rands], s=5)
                # plt.savefig(f"{savepath}/region_{n}_objects.png")
                # plt.clf()
               
                # check for region crossing the RA = 180 deg line
                if abs(highra - lowra) > 180*u.deg:
                    print(f"Region {n} crosses RA=180 deg line. Adjusting bounds.")
                    ra_wrapped[ra_wrapped < 0*u.deg] += 360*u.deg
                    lowra, highra = (
                        ra_wrapped.min() - (cutout_rad_deg+0.5*u.deg),
                        ra_wrapped.max() + (cutout_rad_deg+0.5*u.deg),
                    )
                    
                # lowra = lowra.wrap_at(360*u.deg).deg
                # highra = highra.wrap_at(360*u.deg).deg
                # lowdec = lowdec.deg
                # highdec = highdec.deg
                    
                print(f"Reading chunk of map with bounds RA: [{lowra:.2f},{highra:.2f}], Dec: [{lowdec:.2f},{highdec:.2f}]")
                if size > 1:
                    imap = enmap.read_map(
                        maps[m]["path"],
                        box=[
                            [np.radians(lowdec.value), np.radians(lowra.value)],
                            [np.radians(highdec.value), np.radians(highra.value)],
                        ],
                    )
            
                for z in np.linspace(
                    zmin, zmax, int((zmax - zmin) / dz_rescale)
                ):  # iterate through small z bins
                    Mpc_per_deg_phys_z = cosmo.kpc_proper_per_arcmin(z).to(
                        u.Mpc / u.degree
                    )
                    Mpc_per_deg_comov_z = cosmo.kpc_comoving_per_arcmin(z).to(
                        u.Mpc / u.degree
                    )
                    phys_rescale_factor = Mpc_per_deg_phys_zmin / Mpc_per_deg_phys_z
                    comov_rescale_factor = Mpc_per_deg_comov_zmin / Mpc_per_deg_comov_z
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
                    if ChunkObj.nObj == 0:
                        # set all arrays as nan
                        stack_n = [np.nan]
                        stack_n_phys = [np.nan]
                        stack_n_comov = [np.nan]
                    else:

                        # get the stack
                        print("Stacking region", n, "at z", z)
                        stack_n, stack_n_phys, stack_n_comov = stackChunk(
                            ChunkObj,
                            imap,
                            cutout_rad_deg.value,
                            cutout_resolution.value,
                            orient=orient,
                            rescale_1=phys_rescale_factor,
                            rescale_2=comov_rescale_factor,
                        )
                    # save to this delta-z subgroup
                    z_group.attrs["Nobj"] = ChunkObj.nObj
                    z_group.create_dataset("stack_deg", data=stack_n)
                    z_group.create_dataset("stack_phys", data=stack_n_phys)
                    z_group.create_dataset("stack_comov", data=stack_n_comov)
                    z_group.create_dataset("RA", data=Cat.RA[in_reg & inz])
                    z_group.create_dataset("dec", data=Cat.DEC[in_reg & inz])
                    z_group.create_dataset("z", data=Cat.Z[in_reg & inz])
                    nobj_regn += ChunkObj.nObj
                reg_group.attrs["Nobj"] = nobj_regn
else:
    assert restart_run, (
        f"File {file_i} already exists. If you want to retry consolidating the files, set restart_run=True."
    )
if use_mpi and size > 1:
    # wait for the others to finish writing
    print("Rank", rank, "waiting for others to finish writing.")
    comm.Barrier()
    # collect all
    if rank == 0 and size > 1:
        print("Consolidating stacks to", outfile)

        with h5py.File(outfile, "w") as consol_f:
            files = glob.glob(f"{savepath}/stacks_{stkpts_str}*{teststr}.h5")
            for m in maps:
                mappath = maps[m]["path"]
                sn = maps[m]["shortname"]
                consol_f.create_group(sn)
                for fname in sorted(files):
                    with h5py.File(fname, "r") as mpif:
                        if (
                            f"_0{teststr}.h5" in fname
                        ):  # if the rank_0 file, copy over the attributes (only need to do once)
                            consol_f.attrs["cutout_rad_deg"] = mpif.attrs[
                                "cutout_rad_deg"
                            ]
                            consol_f.attrs["cutout_rad_cMpc"] = mpif.attrs[
                                "cutout_rad_cMpc"
                            ]
                            consol_f.attrs["cutout_rad_pMpc"] = mpif.attrs[
                                "cutout_rad_pMpc"
                            ]
                            consol_f[sn].attrs["map_path"] = mpif[sn].attrs["map_path"]
                        for group in mpif[sn].keys():
                            mpif[sn].copy(mpif[sn][group], consol_f[sn], name=group)
            for file in files:
                print(f"Removing {file}")
                os.remove(file)
        print(f"Saved to {outfile}")
else:
    os.rename(f"{savepath}/stacks_{stkpts_str}_{rank}{teststr}.h5", outfile)
    print(f"Saved to {outfile}")
