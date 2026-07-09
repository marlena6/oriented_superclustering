import sys
# insert path 1 level up
#sys.path.insert(0, "/global/cfs/cdirs/act/data/mlokken/oriented_stacks/oriented_superclustering/")
sys.path.insert(0, "/global/homes/b/boryanah/repos/oriented_superclustering")
import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from pixell import enmap, reproject, utils, curvedsky as cs
import catalog
from kmeans_radec import kmeans_sample
from stacking_functions import Chunk, stackChunk, StackGeometry, extractThumbnails
import h5py
from pathlib import Path
import os
import shutil
import yaml
import glob
import filecmp
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import time
import healpy as hp

start = time.time()
# Load config
if len(sys.argv) != 2:
    raise ValueError("Please provide a config yaml file as an argument.")
config_file_path = sys.argv[1]
print(f"Loading config from {config_file_path}")
with open(config_file_path, "r") as f:
    cfg = yaml.safe_load(f)

use_mpi = cfg["mpi"]["use_mpi"]
if use_mpi:
    from mpi4py import MPI

    # get the MPI ingredients
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm.Barrier()
else:
    rank = 0
    size = 1

print("rank", rank, "passed barrier")

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
# make sure errors = False corresponds to size = 1 and errors = True corresponds to nreg > 1
if errors:
    nreg = cfg["errors"]["nreg"]
    assert nreg > 1, "nreg must be > 1 when errors are enabled."
else:
    nreg = 1
    assert size == 1, "MPI size must be 1 when errors are disabled."

orient = cfg["analysis"]["orient"]
cutout_rad = cfg["analysis"]["cutout_rad_mpc"] * u.Mpc
dz_rescale = cfg["analysis"]["dz_rescale"]
zmin = cfg["analysis"]["zmin"]
zmax = cfg["analysis"]["zmax"]
nu_min = cfg["analysis"]["nu_min"]
nu_max = cfg["analysis"]["nu_max"]
e_min = cfg["analysis"]["e_min"]
e_max = cfg["analysis"]["e_max"]

if zmin in ["None","none",None, ""]:
    zmin = None
if zmax in ["None","none",None, ""]:
    zmax = None
basepath = cfg["paths"]["basepath"]
orientfile = cfg["paths"]["orient_file"]
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

map = cfg["map"]


# check if each map is an enmap, and convert from healpix to enmap otherwise
# does another map exist with _enmap in the same directory?
if rank == 0:
    mappath = map["path"]
    if not os.path.exists(mappath):
        raise ValueError(f"Map path {mappath} does not exist.")
    if map["type"] == 'y':
        try:
            shape, wcs = enmap.read_map_geometry(mappath) # quick geometry check which will fail if not an enmap
            print("Map is already an enmap.")
        except Exception as e:
            print("Map is not an enmap. Attempting to convert from healpix.")
            enmap_path = mappath.replace(".fits", "_enmap.fits")
            if os.path.exists(enmap_path):
                print("Enmap version of map already exists at {enmap_path}. Using that.")
                map["path"] = enmap_path
            else:
                import healpy as hp
                # get the nside without loading the full healpix map
                
                hpmap = hp.read_map(mappath)
                print("Reprojecting healpix map to enmap. This may take a while...")
                shape,wcs = enmap.fullsky_geometry(res=1 * utils.arcmin,proj='car') # 1 arcmin pixels
                enmap_rp = reproject.healpix2map(hpmap, shape=shape, wcs=wcs, method='spline')
                enmap.write_map(enmap_path, enmap_rp)
                print(f"Saved enmap version of to {enmap_path}.")
            map["path"] = enmap_path
    elif map["type"] == 'kappa':
        try:
            shape, wcs = enmap.read_map_geometry(mappath) # quick geometry check which will fail if not an enmap
            print("Map is already an enmap.")
        except Exception as e:
            print("Map is not an enmap. Attempting to convert alms to map.")
            kappa = hp.read_alm(mappath)
            kappa = np.nan_to_num(kappa, nan=0.0) # turn nans to zeros, they are causing troubles
            # use the geometry of the ACT ymap
            shape,wcs = enmap.read_map_geometry("/global/cfs/projectdirs/act/www/dr6_nilc/ymaps_20230220/ilc_actplanck_ymap.fits")
            filter = np.loadtxt("/global/cfs/projectdirs/act/www/dr6_lensing_v1/maps/baseline/kappa_filter_act_dr6_lensing_v1_baseline.txt")
            filter_kappa = hp.almxfl(kappa, filter[:,1])
            kappamap = cs.alm2map(filter_kappa, enmap.empty(shape, wcs))
            enmap_path = mappath.replace(".fits", "_enmap.fits")
            enmap_path = os.path.join(map["savemap_path"], os.path.basename(enmap_path))
            enmap.write_map(enmap_path, kappamap)
            print("Kappa map written at", enmap_path)
            # no nans
            assert(not np.any(np.isnan(kappamap))), "Error: NaNs found in the kappa map"
            map["path"] = enmap_path
        

if use_mpi and size > 1:
    # make sure the mappaths are consistent in case they got changed in the previous step
    if rank==0:
        mappath = map["path"]
    else:
        mappath = None
    path = comm.bcast(mappath, root=0)
    map["path"] = path

# make sure mappath updated
mappath = map["path"]
print("mappath now", mappath)
if not os.path.exists(mappath):
    raise ValueError(f"Map path {mappath} does not exist.")
# read the orientation information
if rank == 0:
    # if not already there, save a copy of the orient file in the new directory for bookkeeping
    if not os.path.exists(savepath + os.path.basename(orientfile)):
        shutil.copy(orientfile, os.path.join(savepath, os.path.basename(orientfile)))
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
        shutil.copy(config_file_path, savepath + "/stacking_config_used.yaml")

config = {'nu_min':nu_min, 'nu_max':nu_max, 'e_min':e_min, 'e_max':e_max}
# every rank reads -- initially had this different but finding issues with broadcasting object on NERSC
cat = catalog.Catalog(
    name="standard",
    pathInCatalog=orientfile,
    nObj=nObj,
    config=config
)
print(f"Rank {rank} read the catalog with {len(cat.Z)} objects.")
if zmin is None:
    zmin = np.amin(cat.Z)
    print(f"zmin not provided. Using minimum redshift in catalog: {zmin:.3f}")
if zmax is None:
    zmax = np.amax(cat.Z)
# make sure the redshift range is reasonable
assert zmin < zmax, f"zmin ({zmin}) must be less than zmax ({zmax})."
assert zmin > 0, f"zmin ({zmin}) must be greater than 0."
assert zmax < 2, f"zmax ({zmax}) must be less than 2."
print(f"Redshift range to stack: {zmin:.3f} - {zmax:.3f}")
print("Map entered:", map)
outfile = (
        f"{savepath}/{map['shortname']}_consol_stacks_z{zmin:.2f}_{zmax:.2f}_{Path(orientfile).stem}{teststr}.h5"
    )

# Make sure the output file doesn't already exist
if rank == 0:
    print("Will save to", outfile)
assert not os.path.exists(outfile), f"Final output file already exists at: {outfile}"

#### getting the region splits for errors ####
# if labels file already exists, read from that
labels_file = f"{savepath}/region_labels_{nreg}reg{teststr}.txt"
if restart_run:
    assert os.path.exists(labels_file), (
        f"Region labels file {labels_file} not found. Cannot restart run without it. Set restart_run=False to generate new region labels."
    )
    
if errors:
    if os.path.exists(labels_file) and restart_run:
        labels = np.loadtxt(labels_file, dtype=np.int64)
        print(f"Read region labels from {labels_file}")
    else:
        if rank == 0:
            km = kmeans_sample(np.vstack((cat.RA, cat.DEC)).T, nreg, maxiter=100, tol=1.0e-5)
            labels = km.labels.astype(np.int64)
            np.savetxt(labels_file, labels, fmt="%d")
            print(f"Saved region labels to {labels_file}")
            colors = ['C'+str(i) for i in range(10)]
            rands  = np.random.choice(np.arange(len(cat.RA)), size=1000, replace=False)
            plt.scatter(cat.RA[rands], cat.DEC[rands], c=[colors[l%10] for l in labels[rands]], s=5)
            plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=f'Region {i}', markerfacecolor=colors[i%10], markersize=5) for i in range(nreg)],loc='best')
            plt.savefig(f"{savepath}/region_splits.png")
            plt.clf()
        else:
            labels = None 

        if use_mpi and size > 1:
            labels = comm.bcast(labels, root=0)
    cat.labels = labels  # add labels to the Catalog object
    print(f"Rank {rank} has labels with unique values: {np.unique(labels)}")
else:
    cat.labels = np.zeros(len(cat.RA), dtype=np.int64) # they are all region '0'
    

# if the unit of cutout_rad is Mpc, then we need to convert it to degrees
if cutout_rad.unit == u.Mpc:
    # find out the stack geometry based on the redshift where the cutout size is largest in angular units
    z_array = np.linspace(zmin, zmax, int((zmax - zmin) / dz_rescale) + 1)
    angular_size_z = cosmo.arcsec_per_kpc_comoving(z_array).to(u.arcmin / u.Mpc) * cutout_rad
    z_largest_cutout = z_array[np.argmax(angular_size_z)]
    print("Cutout size is largest at z =", z_largest_cutout)
    Mpc_per_deg_comov_base = cosmo.kpc_comoving_per_arcmin(z_largest_cutout).to(u.Mpc / u.degree)
    Mpc_per_deg_phys_base = cosmo.kpc_proper_per_arcmin(z_largest_cutout).to(u.Mpc / u.degree)
    
    cutout_rad_deg = (
        1 / (cosmo.kpc_comoving_per_arcmin(z_largest_cutout).to(u.Mpc / u.deg)) * cutout_rad
    )
elif cutout_rad.unit in [u.arcmin, u.arcsec]:
    cutout_rad_deg = cutout_rad.to(u.deg)
else:
    raise ValueError(
        "cutout_rad must be in units of Mpc, degrees, arcminutes, or arcseconds"
    )

cutout_resolution_deg = (0.5 * u.arcmin).to(u.deg)
print(
    f"will take thumbnails with size {cutout_rad_deg:.2f} and resolution {cutout_resolution_deg:.2f}."
)

# get the stack geometry just once
geom = StackGeometry(cutout_rad_deg.value, cutout_resolution_deg.value)

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


end = time.time()
print("Whole setup took", end - start, "seconds.")

# Prepare to save to an HDF5 file
file_i = f"{savepath}/stacks_{Path(orientfile).stem}_{rank}{teststr}.h5"
if not os.path.exists(file_i):
    with h5py.File(file_i, "w") as f:
        f.attrs["cutout_rad_deg"] = cutout_rad_deg.value
        f.attrs["cutout_rad_cMpc"] = cutout_rad_deg * Mpc_per_deg_comov_base.value
        f.attrs["cutout_rad_pMpc"] = cutout_rad_deg * Mpc_per_deg_phys_base.value
        if cat.hdr is not None:
            f.attrs["orientation_constraints"] = cat.hdr
        elif cat.constraints is not None:
            f.attrs["orientation_constraints"] = np.unique(cat.constraints).tolist()
            
    
        mappath = map["path"]
        sn = map["shortname"]
        map_group = f.create_group(sn)
        map_group.attrs["map_path"] = mappath
        print(f"Handling input map: {mappath}")
        if size==1:
            readmap_start = time.time()
            # read the whole map
            imap = enmap.read_map(mappath)
            readmap_end = time.time()
            print(f"Read map in {readmap_end - readmap_start:.1f} seconds.")
        for i in range(nruns_local + extras):
            n = rank * nruns_local + i
            in_reg = cat.labels == n
            print("in_reg type", type(in_reg))
            print("in_reg shape", in_reg.shape)
            print(f"Rank {rank}, region {n}, Nobj = {in_reg.sum()}")
            nobj_regn = 0
            # make an HDF5 group for this region
            reg_group = map_group.create_group(f"reg_{n}")
            reg_group.attrs["Region"] = n
            print(f"Analyzing region {n}")
            # define the map edges for this region
            sc = SkyCoord(ra=cat.RA[in_reg]*u.deg, dec=cat.DEC[in_reg]*u.deg, frame="icrs")
            ra_wrapped = sc.ra.wrap_at(180*u.deg)
            lowra, highra = (
                ra_wrapped.min() - (cutout_rad_deg+0.5*u.deg),
                ra_wrapped.max() + (cutout_rad_deg+0.5*u.deg),
            )
            lowdec, highdec = (
                sc.dec.min() - (cutout_rad_deg+0.5*u.deg),
                sc.dec.max() + (cutout_rad_deg+0.5*u.deg),
            )
            
            if size > 1:
                # check for region crossing the RA = 180 deg line
                if abs(highra - lowra) > 180*u.deg:
                    print(f"Region {n} crosses RA=180 deg line. Adjusting bounds.")
                    ra_wrapped[ra_wrapped < 0*u.deg] += 360*u.deg
                    lowra, highra = (
                        ra_wrapped.min() - (cutout_rad_deg+0.5*u.deg),
                        ra_wrapped.max() + (cutout_rad_deg+0.5*u.deg),
                    )
                    
                print(f"Reading chunk of map with bounds RA: [{lowra:.2f},{highra:.2f}], Dec: [{lowdec:.2f},{highdec:.2f}]")
                imap = enmap.read_map(
                    mappath,
                    box=[
                        [np.radians(lowdec.value), np.radians(highra.value)],
                        [np.radians(highdec.value), np.radians(lowra.value)],
                    ],
                )
            # extract all the thumbnails for this region
            alpha_inreg = cat.alpha[in_reg] if cat.alpha is not None else None
            x_asym_inreg = cat.x_asym[in_reg] if cat.x_asym is not None else None
            y_asym_inreg = cat.y_asym[in_reg] if cat.y_asym is not None else None
            ra_inreg = cat.RA[in_reg]
            dec_inreg = cat.DEC[in_reg]
            z_inreg = cat.Z[in_reg]
            # B.H.
            vr_inreg = cat.vR[in_reg] if cat.vR is not None else None
            chunkObj_reg = Chunk( # B.H. vR flag in the yaml file
                    ra_inreg,
                    dec_inreg,
                    alpha_inreg,
                    x_asym_inreg,
                    y_asym_inreg,
                    vr_inreg - np.mean(vr_inreg) # B.H.
                )
            thumbs_time = time.time()
            thumbs = extractThumbnails(
                chunkObj_reg,
                geom,
                imap,
                orient
            )
            thumbs_time_end = time.time()
            print(f"Extracted thumbnails for region {n} in {thumbs_time_end - thumbs_time:.1f} seconds.")
            
            stacking_start = time.time()
            
            for i in range(len(z_array)-1): # iterate through small z bins
                z = z_array[i] # just use the lower z of this slice
                Mpc_per_deg_phys_z = cosmo.kpc_proper_per_arcmin(z).to(
                    u.Mpc / u.degree
                )
                Mpc_per_deg_comov_z = cosmo.kpc_comoving_per_arcmin(z).to(
                    u.Mpc / u.degree
                )
                phys_rescale_factor = Mpc_per_deg_phys_base / Mpc_per_deg_phys_z
                comov_rescale_factor = Mpc_per_deg_comov_base / Mpc_per_deg_comov_z
                inz = (cat.Z[in_reg] < (z_array[i+1])) & (cat.Z[in_reg] > (z_array[i]))
                z_rescale_str = f"z_{z_array[i]:.2f}_{z_array[i+1]:.2f}"
                z_group = reg_group.create_group(z_rescale_str)  # create a subgroup
                print("Creating z group for z range", z_array[i], "-", z_array[i+1], "with", inz.sum(), "objects.")
                # make the ChunkObj for these z
                
                if alpha_inreg is not None:
                    alpha_inreg_inz = alpha_inreg[inz]
                else:
                    alpha_inreg_inz = None
                if x_asym_inreg is not None:
                    x_asym_inreg_inz = x_asym_inreg[inz]
                else:
                    x_asym_inreg_inz = None
                if y_asym_inreg is not None:
                    y_asym_inreg_inz = y_asym_inreg[inz]
                else:
                    y_asym_inreg_inz = None
                if vr_inreg is not None:
                    vr_inreg_inz = vr_inreg[inz]
                else:
                    vr_inreg_inz = None
                chunkObj = Chunk( # B.H.
                    ra_inreg[inz],
                    dec_inreg[inz],
                    alpha_inreg_inz,
                    x_asym_inreg_inz,
                    y_asym_inreg_inz,
                    vr_inreg_inz - np.mean(vr_inreg_inz)
                )
                
                if chunkObj.nObj == 0:
                    # set all arrays as nan
                    stack_n = [np.nan]
                    stack_n_phys = [np.nan]
                    stack_n_comov = [np.nan]
                else:
                    # get the thumbs for these z
                    thumbs_inz = thumbs[inz]
                    # get the stack
                    print("Stacking region", n, "at z", z)
                    stack_n, stack_n_phys, stack_n_comov = stackChunk(
                        chunkObj,
                        geom,
                        imap,
                        orient=orient,
                        rescale_1=phys_rescale_factor,
                        rescale_2=comov_rescale_factor,
                        thumbnails=thumbs_inz 
                    )
                # save to this delta-z subgroup
                z_group.attrs["Nobj"] = chunkObj.nObj
                z_group.create_dataset("stack_deg", data=stack_n)
                z_group.create_dataset("stack_phys", data=stack_n_phys)
                z_group.create_dataset("stack_comov", data=stack_n_comov)
                z_group.create_dataset("RA", data=ra_inreg[inz])
                z_group.create_dataset("dec", data=dec_inreg[inz])
                z_group.create_dataset("z", data=z_inreg[inz])
                nobj_regn += chunkObj.nObj
            reg_group.attrs["Nobj"] = nobj_regn
            stacking_end = time.time()
            print(f"Finished stacking region {n} in {stacking_end - stacking_start:.1f} seconds.")
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
            files = glob.glob(f"{savepath}/stacks_{Path(orientfile).stem}*{teststr}.h5")
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
    os.rename(f"{savepath}/stacks_{Path(orientfile).stem}_{rank}{teststr}.h5", outfile)
    print(f"Saved to {outfile}")
