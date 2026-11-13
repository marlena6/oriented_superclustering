import sys
import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from pixell import enmap, reproject, utils
import catalog
from kmeans_radec import kmeans_sample
from stacking_functions import Chunk, stackChunk, StackGeometry, extractThumbnails, readmap
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
import healpy as h
from utils import dist_to_nearest_edge


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

newdir_name = cfg["run"]["newdir_name"]
test = cfg["run"]["test"]
if test:
    nObj = cfg["run"]["nObj_test"]
    teststr = f"_test{nObj:.1e}"
    print("nObj_test is set to", nObj, "for testing purposes.")
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
avoid_mask_by = cfg["analysis"]["avoid_mask_by"] # degrees from which to avoid mask edges

if zmin in ["None","none",None, ""]:
    zmin = None
if zmax in ["None","none",None, ""]:
    zmax = None
basepath = cfg["paths"]["basepath"]
orientfile = cfg["paths"]["orient_file"]
maskfile_list = cfg["paths"]["mask_file_list"]
inmap_info = cfg["map"]

# multiple mask files can be entered. If multiple, try to multiply them together.
savepath = os.path.join(basepath, newdir_name)


# have rank 0 make the new directory, all others wait
if rank == 0:
    assert not os.path.exists(savepath), (
        f"Directory {savepath} already exists. Please delete it or set a different newdir_name in the config file."
    )
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        print(f"Created directory {savepath} for this run.")
    
    inmap_info = readmap(inmap_info) # reads the map, filters / reprojects if necessary, modifies path in inmap_info if necessary
    
    
        
        
if use_mpi and size > 1:
    comm.Barrier()  # wait for rank 0 to finish making the directory
    # make sure the mappaths are consistent in case they got changed in the previous step
    if rank==0:
        mappath = inmap_info["path"]
    else:
        mappath = None
    path = comm.bcast(mappath, root=0)
    inmap_info["path"] = path

# make sure mappath updated
mappath = inmap_info["path"]
print("mappath now", mappath)
if not os.path.exists(mappath):
    raise ValueError(f"Map path {mappath} does not exist.")

# Reading the geometry of the bigger map to keep for later
shape,wcs=enmap.read_fits_geometry(mappath, hdu=None, quick=True)
map_box_deg = np.degrees(enmap.box(shape, wcs))   # shape (2,2): [[dec_min,ra_min],[dec_max,ra_max]] in radians
map_dec_min, map_dec_max = np.sort(map_box_deg[:, 0])
map_ra_min,  map_ra_max  = np.sort(map_box_deg[:, 1])   # sort: CAR RA usually decreases with pixel index
map_full_ra = (map_ra_max - map_ra_min) >= 359.9        # full-sky-in-RA maps have no RA edge

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

## make sure same selection of Catalog is done across ranks, if in test mode ##
seed = None
if nObj is not None:
    seed_file = f"{savepath}/catalog_seed{teststr}.txt"
    if rank == 0:
        if cfg["run"].get("seed") is not None:
            seed = int(cfg["run"]["seed"])  # optional manual override
        else:
            seed = int(np.random.SeedSequence().entropy % 2**63) # fresh OS entropy
        np.savetxt(seed_file, [seed], fmt="%d")
    if use_mpi and size > 1:
        seed = comm.bcast(seed, root=0)
# every rank reads -- initially had this different but finding issues with broadcasting object on NERSC
cat = catalog.Catalog(
    pathInCatalog=orientfile,
    nObj=nObj,
    config=config,
    seed=seed
)

print(f"Rank {rank} read the catalog with {len(cat.z)} objects.")
velocity_stack = bool(cfg["analysis"].get("velocity_stack", False))
#if "velocity_stack" is True, cat.vr must be provided
if velocity_stack:
    assert cat.vr is not None, "Need to provide vr in catalog if you want to do velocity-weighted stacking. Otherwise set velocity_stack to False in the config."
if zmin is None:
    zmin = np.amin(cat.z)
    print(f"zmin not provided. Using minimum redshift in catalog: {zmin:.3f}")
if zmax is None:
    zmax = np.amax(cat.z)
# make sure the redshift range is reasonable
assert zmin < zmax, f"zmin ({zmin}) must be less than zmax ({zmax})."
assert zmin > 0, f"zmin ({zmin}) must be greater than 0."
assert zmax < 2, f"zmax ({zmax}) must be less than 2."
print(f"Redshift range to stack: {zmin:.3f} - {zmax:.3f}")
print("Map entered:", inmap_info)
outfile = (
        f"{savepath}/{inmap_info['shortname']}_consol_stacks_z{zmin:.2f}_{zmax:.2f}_{Path(orientfile).stem}{teststr}.h5"
    )

# Make sure the output file doesn't already exist
if rank == 0:
    print("Will save to", outfile)
assert not os.path.exists(outfile), f"Final output file already exists at: {outfile}"

#### getting the region splits for errors ####
labels_file = f"{savepath}/region_labels_{nreg}reg{teststr}.txt"


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

pad_deg = 0.0 if orient=='original' else 0.2
r_need = ((cutout_rad_deg.value + pad_deg) * np.sqrt(2)) * u.deg   # deg
margin = 0.1 * u.deg

maskpath=None
if rank==0:
    if maskfile_list is not None:
        print("Found masks")
        if len(maskfile_list)==1 and "shrunk" in maskfile_list[0]:
            maskpath = maskfile_list[0]
            combined_mask = enmap.read_map(maskpath)
        else:
            for i, maskpath in enumerate(maskfile_list):
                maskpath = readmap({"path":maskpath,"type":"mask"})  # readmap will filter / reproject if necessary
                # try:
                #     mask = enmap.read_map(maskpath)
                mask = enmap.read_map(maskpath["path"])
                if i == 0:
                    combined_mask = mask
                else:
                    combined_mask *= mask
                # except Exception as e:
                #     raise ValueError(f"Could not read mask file {maskpath}. Please ensure it is enmap format.") from e

            # shrink the True part of the mask the amount given, or by size of the cutouts
            print("Shrinking mask...")
            if avoid_mask_by is not None:
                combined_mask = enmap.shrink_mask(combined_mask, (avoid_mask_by*u.deg).to(u.rad).value)
            else:
                combined_mask = enmap.shrink_mask(combined_mask, r_need.to(u.rad).value)
            # write mask to new file and delete
            maskpath = os.path.join(savepath,"mask_shrunk.fits")
            # Ensure FITS-compatible binary mask
            combined_mask = (combined_mask > 0).astype(np.uint8)
            enmap.write_map(maskpath, combined_mask) # error here ML
            print("Combined mask written to", maskpath)
        # rank 1 reduces its catalog
        cat.mask_catalog(combined_mask)
        del combined_mask

# cut catalog if map is not full-sky
r = (r_need + margin).to(u.deg).value
dec = cat.dec
ok = (dec - map_dec_min >= r) & (map_dec_max - dec >= r)
if not map_full_ra:
    ra_w = (cat.ra - map_ra_min) % 360 + map_ra_min            # put catalog RA in the map's RA convention
    cosd = np.cos(np.radians(np.minimum(np.abs(dec) + r, 90)))  # worst-case cos(dec) across the thumbnail
    ok &= (cosd * (ra_w - map_ra_min) >= r) & (cosd * (map_ra_max - ra_w) >= r)
print(f"Rank {rank}: map-edge cut keeps {ok.sum()}/{len(ok)} objects")
cat.prune_catalog(index=ok, inplace=True)

# share new maskpath with the other ranks
if use_mpi and size > 1:
    maskpath = comm.bcast(maskpath, root=0)

# every other rank now reads in the mask
if rank > 0:
    if maskpath is not None:
        combined_mask = enmap.read_map(maskpath)
        cat.mask_catalog(combined_mask)
        del combined_mask
    
if errors:
    
    if rank == 0:
        print("N regions", nreg)
        print(len(cat.ra))
        print(len(cat.dec))
        km = kmeans_sample(np.vstack((cat.ra, cat.dec)).T, nreg, maxiter=100, tol=1.0e-5)
        labels = km.labels.astype(np.int64)
        np.savetxt(labels_file, labels, fmt="%d")
        print(f"Saved region labels to {labels_file}")
        colors = ['C'+str(i) for i in range(10)]
        rands  = np.random.choice(np.arange(len(cat.ra)), size=1000, replace=False)
        plt.scatter(cat.ra[rands], cat.dec[rands], c=[colors[l%10] for l in labels[rands]], s=5)
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
    cat.labels = np.zeros(len(cat.ra), dtype=np.int64) # they are all region '0'
    


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
            print("Header of input catalog:", cat.hdr)
            f.attrs["orientation_constraints"] = "\n".join(cat.hdr.tolist())

    
        mappath = inmap_info["path"]
        sn = inmap_info["shortname"]
        map_group = f.create_group(sn)
        map_group.attrs["map_path"] = mappath
        print(f"Handling input map: {mappath}")
        min_safe_dist_rad = r_need.to(u.rad).value 

        if size==1:
            readmap_start = time.time()
            # read the whole map
            imap = enmap.read_map(mappath)
            if maskfile_list is not None:
                imask = enmap.read_map(maskpath)
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
            sc = SkyCoord(ra=cat.ra[in_reg]*u.deg, dec=cat.dec[in_reg]*u.deg, frame="icrs")
            ra_wrapped = sc.ra.wrap_at(180*u.deg)
            lowdec  = max(sc.dec.min() - r_need - margin, -90 * u.deg)
            highdec = min(sc.dec.max() + r_need + margin,  90 * u.deg)
            cosd    = np.cos(max(abs(lowdec), abs(highdec)))   # worst-case cos(dec) inside the box
            ra_pad  = r_need / cosd + margin
            lowra, highra = (
                ra_wrapped.min() - ra_pad,
                ra_wrapped.max() + ra_pad,
            )
            
            if size > 1:
                # check for region crossing the ra = 180 deg line
                if abs(highra - lowra) > 180*u.deg:
                    print(f"Region {n} crosses RA=180 deg line. Adjusting bounds.")
                    ra_wrapped[ra_wrapped < 0*u.deg] += 360*u.deg
                    lowra, highra = (
                        ra_wrapped.min() - ra_pad,
                        ra_wrapped.max() + ra_pad,
                    )
                    
                print(f"Reading chunk of map with bounds RA: [{lowra:.2f},{highra:.2f}], Dec: [{lowdec:.2f},{highdec:.2f}]")
                box = [
                        [np.radians(lowdec.value), np.radians(highra.value)],
                        [np.radians(highdec.value), np.radians(lowra.value)],
                    ]
                imap = enmap.read_map(mappath, box=box)
                if maskfile_list is not None:
                    imask = enmap.read_map(maskpath, box=box)

            # extract the points within the sky region
            
            alpha_inreg = cat.alpha[in_reg] if cat.alpha is not None else None
            x_pol_inreg = cat.x_pol[in_reg] if cat.x_pol is not None else None
            y_pol_inreg = cat.y_pol[in_reg] if cat.y_pol is not None else None
            ra_inreg = ra_wrapped.degree
            dec_inreg = cat.dec[in_reg]
            z_inreg = cat.z[in_reg]
            # B.H.
            

            vr_inreg = cat.vr[in_reg] if cat.vr is not None else None
            # check if velocity_stack is True or False if vr is not None

            ## Distance computed
            dist_imap = dist_to_nearest_edge(np.radians(dec_inreg), np.radians(ra_inreg), np.radians(lowdec.value), np.radians(highdec.value), np.radians(lowra.value), np.radians(highra.value))
            # print("RA bounds:", lowra, highra)
            # print("RA range of objects:", ra_inreg.min(), ra_inreg.max())
            # print("dist_imap min/max:", np.degrees(dist_imap).min(),
                                        # np.degrees(dist_imap).max())

            edge_ok  = dist_imap >= min_safe_dist_rad #RH checking all distances
            assert edge_ok.all(), "Some objects passing the map boundaries"

            if test:
                plt.hist(dist_imap, bins=6, color="steelblue",
                        edgecolor="white", linewidth=0.6, alpha=0.85)
                plt.savefig(f"{savepath}/region_{n}_local_diffs.png")
            
            
            # extract all the thumbnails for this region that have ok distances
            alpha_inreg = alpha_inreg[edge_ok] if cat.alpha is not None else None
            x_pol_inreg = x_pol_inreg[edge_ok] if cat.x_pol is not None else None
            y_pol_inreg = y_pol_inreg[edge_ok] if cat.y_pol is not None else None
            ra_inreg = ra_inreg[edge_ok]
            dec_inreg = dec_inreg[edge_ok]
            z_inreg = z_inreg[edge_ok]
            vr_inreg = vr_inreg[edge_ok] if cat.vr is not None else None
            
            diff_lowra_inreg= (ra_inreg - lowra.value)
            diff_highra_inreg = (highra.value-ra_inreg )
            diff_lowdec_inreg= (dec_inreg - lowdec.value)
            diff_highdec_inreg = (highdec.value-dec_inreg)

            ## Distance computed again for check
            dist_imap = dist_to_nearest_edge(np.radians(dec_inreg), np.radians(ra_inreg), np.radians(lowdec.value), np.radians(highdec.value), np.radians(lowra.value), np.radians(highra.value))
    
            if test:
                plt.hist(dist_imap, bins=6, color="steelblue",
                        edgecolor="white", linewidth=0.6, alpha=0.85)
                plt.savefig(f"{savepath}/region_{n}_local_diffs_postcut.png")

            chunkObj_reg = Chunk(
                    ra_inreg,
                    dec_inreg
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
                inz = (z_inreg < (z_array[i+1])) & (z_inreg > (z_array[i]))
                z_rescale_str = f"z_{z_array[i]:.2f}_{z_array[i+1]:.2f}"
                z_group = reg_group.create_group(z_rescale_str)  # create a subgroup
                print(f"Creating z group for z range {z_array[i]:.2f}-{z_array[i+1]:.2f} with {inz.sum()} objects.")
                # make the ChunkObj for these z
                
                if alpha_inreg is not None:
                    alpha_inreg_inz = alpha_inreg[inz]
                else:
                    alpha_inreg_inz = None
                if x_pol_inreg is not None:
                    x_pol_inreg_inz = x_pol_inreg[inz]
                else:
                    x_pol_inreg_inz = None
                if y_pol_inreg is not None:
                    y_pol_inreg_inz = y_pol_inreg[inz]
                else:
                    y_pol_inreg_inz = None
                if vr_inreg is not None:
                    vr_inreg_inz = vr_inreg[inz] - np.mean(vr_inreg[inz])
                else:
                    vr_inreg_inz = None
                chunkObj = Chunk( # B.H.
                    ra_inreg[inz],
                    dec_inreg[inz],
                    alpha_inreg_inz,
                    x_pol_inreg_inz,
                    y_pol_inreg_inz,
                    vr_inreg_inz,
                    velocity_stack=velocity_stack
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
                        imap=imap,
                        orient=orient,
                        rescale_1=phys_rescale_factor,
                        rescale_2=comov_rescale_factor,
                        thumbnails=thumbs_inz,
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
    sys.exit(f"File {file_i} already exists. Please delete it or set a different newdir_name in the config file.")

if use_mpi and size > 1:
    # wait for the others to finish writing
    print("Rank", rank, "waiting for others to finish writing.")
    comm.Barrier()
    print("Rank", rank, "passed barrier.")
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
