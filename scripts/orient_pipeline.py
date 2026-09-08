import numpy as np
import os
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo, z_at_value
import time
import healpy as hp
import sys
import select_and_orient as sao
import yaml
import pandas as pd
import shutil
from catalog import Catalog

def fmt(x, ndec=1):
    return str(int(x)) if x.is_integer() else f"{x:.{ndec}f}".replace('.', 'pt')


h = (cosmo.H0/100.).value


start = time.time()
##################################
# Load config
if len(sys.argv) != 2:
    raise ValueError("Please provide a config yaml file as an argument.")
config_file_path = sys.argv[1]
print(f"Loading config from {config_file_path}")
with open(config_file_path, "r") as f:
    cfg = yaml.safe_load(f)

save_path = cfg["run"]["save_path"]
write_maps = cfg["run"]["write_maps_to_file"] # boolean
filenames_in_Mpc = cfg["run"]["filenames_in_Mpc"] # if True, the filenames will be in Mpc, if False, they will be in z

stack_catalog = cfg["files"]["stacking_object_catalog"]
orient_catalog  = cfg["files"]["orient_object_catalog"]
randoms_catalog = cfg["files"]["randoms_catalog"]
maskfile = cfg["files"]["mask"] # if not None, should be a binary mask: fits file with 1s in the area to use and 0s in the area to mask out

# Mode for splitting the data along the line-of-sight: either custom_zlist, custom_dlist, auto_all, or auto_overlap (automatic bin sandwiching with predefined relative sizes)
los_split_mode = cfg["analysis"]["los_split_mode"]
so_width = cfg["analysis"]["stacking_slice_width_Mpc"] # width of the slice in which to select the stacking objects, in cMpc
oo_width = cfg["analysis"]["orientation_slice_width_Mpc"] # width of the slice in which to select the orientation objects, in cMpc; only relevant if los_split_mode is 'auto_overlap'
# split if you want to only use some of the galaxy data to orient and other to stack
frac_use = cfg["analysis"]["fraction_input_data"]
# Smooth the maps by a Gaussian with this beam FWHM
smth     = cfg["analysis"]["smoothing_Mpc"]
orient_mode = cfg["analysis"]["orientation_mode"] # "original", "random", "sym", "asym_x", "asym_y", "asym_xy"
center_objects_label = cfg["analysis"]["center_objects_label"] # e.g., 'lrgc_dr1' for 'clustering' LRGs
orient_objects_label = cfg["analysis"]["orient_objects_label"] # e.g., 'elgc+lrgc_dr1' for 'clustering' LRGs + ELGs
nside = cfg["analysis"]["healpix_nside"]
minz = cfg["analysis"]["z_min"]
maxz = cfg["analysis"]["z_max"]
##################################


smth_str = ("{:.1f}".format(smth)).replace('.','pt')

pct = frac_use*100
if frac_use != 1:
    pctstr = "_{:.0f}pct".format(pct)
else:
    pctstr = ""

# load the center objects data
cat_so = Catalog(stack_catalog)
# figure out minz and maxz if None
if minz is None:
    minz = cat_so.z.min()
if maxz is None:
    maxz = cat_so.z.max()

mask = None
if maskfile is not None:
    mask = hp.read_map(maskfile)

zstr = "{:.2f}_{:.2f}".format(minz, maxz).replace('.','pt')
# savename for file
save_file = os.path.join(save_path, f"{orient_objects_label}{pctstr}_{zstr}_{smth_str}Mpc_{orient_mode}.csv")
if os.path.exists(save_file):
    raise ValueError(f"Output file {save_file} already exists. Please change the save_path or delete the existing file to avoid overwriting.")
else:
    print("Will save output to", save_file)

if not os.path.exists(save_path):
    os.mkdir(save_path)
    print(f"Created directory {save_path} for this run.")

# copy the config file into the output directory, if not already there
if not os.path.exists(os.path.join(save_path, os.path.basename(config_file_path))):
    shutil.copy(config_file_path, os.path.join(save_path, os.path.basename(config_file_path)))


if los_split_mode == 'custom_zlist':
    # enter the pre-defined redshift bins as the first argument
    zbins = cfg["analysis"]["z_bins"]
    dlist_tot_oo, zlist_tot_oo = sao.dlist(cosmo, zlist=zbins)

elif los_split_mode == 'custom_dlist':
    # enter the pre-defined comoving distance bins as the first argument
    dbins = cfg["analysis"]["chi_bins"]
    dlist_tot_oo, zlist_tot_oo = sao.dlist(cosmo, dlist=dbins)
    
elif los_split_mode == 'auto_overlap':
    minz = minz+.005 # add small buffer
    maxz = maxz-.005 # add small buffer
    comoving_oo_narrowbin_start  = cosmo.comoving_distance(minz).to(u.Mpc)
    comoving_oo_narrowbin_0 = np.array([comoving_oo_narrowbin_start.value, (comoving_oo_narrowbin_start+oo_width*u.Mpc).value])
    nbins = int((cosmo.comoving_distance(maxz).to(u.Mpc).value - comoving_oo_narrowbin_start.value)/so_width)
    dbins = [comoving_oo_narrowbin_0 + i * so_width for i in range(nbins)] 
    print("Have you made sure to customize the width of the orientation and stacking slices? It is currently set to {:d} and {:d} Mpc with a mininum z of {:.2f} and maximum z of {:.2f}.".format(oo_width, so_width, minz, maxz))
    dlist_tot_oo, zlist_tot_oo = sao.dlist(cosmo, dlist=dbins)    

# save the zlist to a file
np.savetxt(os.path.join(save_path, "zlist.txt"), zlist_tot_oo)


# if different, load the orientation data
if orient_catalog is not None:
    cat_oo = Catalog(orient_catalog)
else:
    # use the stack object catalog again
    cat_oo = Catalog(stack_catalog)
# load randoms data
if randoms_catalog is not None:
    cat_ran = Catalog(randoms_catalog)
    
# take a fraction of the orient + randoms catalog if frac is less than 1
if frac_use < 1:
    rand_idx_oo = np.random.choice(len(cat_oo.ra), size=int(frac_use*len(cat_oo.ra)), replace=False)
    cat_oo.prune_catalog(index=rand_idx_oo)
    rand_idx_rand = np.random.choice(len(cat_ran.ra), size=int(frac_use*len(cat_ran.ra)), replace=False)
    cat_ran.prune_catalog(index=rand_idx_rand)

# prune all catalogs to z range and order by z
print(f"Pruning catalogs to z range and sorting by z. Catalog is initially, {len(cat_so.ra)} long.")
cat_so.prune_catalog(condition={"z":(minz, maxz)}, inplace=True)
print("After pruning, catalog is now, {:d} long.".format(len(cat_so.ra)))
cat_oo.prune_catalog(condition={"z":(minz, maxz)}, inplace=True)
cat_so.sort_catalog(sort_by="z")
cat_oo.sort_catalog(sort_by="z")
if randoms_catalog is not None:
    cat_ran.prune_catalog(condition={"z":(minz, maxz)}, inplace=True)
    cat_ran.sort_catalog(sort_by="z")


# set the division of catalogs
dbincent = [(dlist_tot_oo[i][0]+dlist_tot_oo[i][1])/2. for i in range(len(dlist_tot_oo))]
zbincent = [z_at_value(cosmo.comoving_distance, dbincent[i]*u.Mpc).value for i in range(len(dbincent))]
smth_arcmin = [(cosmo.arcsec_per_kpc_comoving(zbincent[i]).to(u.arcmin/u.Mpc) * (smth*u.Mpc)).value for i in range(len(zbincent))]
if los_split_mode=='auto_overlap' or los_split_mode=='custom_dlist':
    # widths of stacking objects and orient objects are different
    dlist_tot_so = np.asarray([[dbincent[i]-so_width/2., dbincent[i]+so_width/2.] for i in range(len(dbincent))])
    zlist_tot_so = np.asarray([[z_at_value(cosmo.comoving_distance, dlist_tot_so[i][0]*u.Mpc).value, z_at_value(cosmo.comoving_distance, dlist_tot_so[i][1]*u.Mpc).value] for i in range(len(dbincent))])
elif los_split_mode=='auto_all' or los_split_mode=='custom_zlist':
    # widths of stacking objects and orient objects are the same
    dlist_tot_so = dlist_tot_oo
    zlist_tot_so = zlist_tot_oo

cat_split_idx_so_lower = np.searchsorted(
    cat_so.z,
    zlist_tot_so[:, 0],
    side='left'
)
cat_split_idx_so_upper = np.searchsorted(
    cat_so.z,
    zlist_tot_so[:, 1],
    side='left'
)
cat_split_idx_oo_lower = np.searchsorted(
    cat_oo.z,
    zlist_tot_oo[:, 0],
    side='left'
)
cat_split_idx_oo_upper = np.searchsorted(
    cat_oo.z,
    zlist_tot_oo[:, 1],
    side='left'
)
cat_split_idx_ran_lower = np.searchsorted(
    cat_ran.z,
    zlist_tot_oo[:, 0],
    side='left'
)
cat_split_idx_ran_upper = np.searchsorted(
    cat_ran.z,
    zlist_tot_oo[:, 1],
    side='left'
)

#### This is where the main calculations happen ####
zloop_begin = time.time()
# add empty-array attributes to the so_cat catalog
cat_so.alpha = np.full(len(cat_so.ra), np.nan)
cat_so.e = np.full(len(cat_so.ra), np.nan)
cat_so.nu = np.full(len(cat_so.ra), np.nan)
cat_so.x_pol = np.full(len(cat_so.ra), np.nan)
cat_so.y_pol = np.full(len(cat_so.ra), np.nan)


for i in range(len(zlist_tot_so)):
    zbin_start_time = time.time()
    print(f"Processing bin {i+1} of {len(zlist_tot_oo)}: Orienting by galaxies from {dlist_tot_oo[i][0]:.1f} to {dlist_tot_oo[i][1]:.1f} Mpc, {zlist_tot_oo[i][0]:.4f} to {zlist_tot_oo[i][1]:.4f} with smoothing scale {smth_arcmin[i]:.2f} arcmin.")
    # find stacking objects only within plus/minus so_width/2 cMpc of the bin center
    print(f"Finding stacking objects from {dlist_tot_so[i][0]:.1f} to {dlist_tot_so[i][1]:.1f} Mpc.")
    print(f"In redshift space, this is between {zlist_tot_so[i][0]:.4f} and {zlist_tot_so[i][1]:.4f}.")
    
    if filenames_in_Mpc:
        binstr_orient = (f"{fmt(dlist_tot_oo[i][0])}_{fmt(dlist_tot_oo[i][1])}Mpc").replace('.','pt')
    else:
        binstr_orient = (f"z{fmt(zlist_tot_oo[i][0])}_{fmt(zlist_tot_oo[i][1])}").replace('.','pt')
    
    # get the orientation objects in the orientation bin
    if randoms_catalog is not None:
        odmap, mask = sao.delta_g(nside, cat_oo.ra[cat_split_idx_oo_lower[i]:cat_split_idx_oo_upper[i]], cat_oo.dec[cat_split_idx_oo_lower[i]:cat_split_idx_oo_upper[i]], ra_rand=cat_ran.ra[cat_split_idx_ran_lower[i]:cat_split_idx_ran_upper[i]], dec_rand=cat_ran.dec[cat_split_idx_ran_lower[i]:cat_split_idx_ran_upper[i]], catalog_weights=cat_oo.w[cat_split_idx_oo_lower[i]:cat_split_idx_oo_upper[i]], randoms_weights=cat_ran.w[cat_split_idx_ran_lower[i]:cat_split_idx_ran_upper[i]], smth=smth_arcmin[i])
    else:
        odmap = sao.delta_g(nside, cat_oo.ra[cat_split_idx_oo_lower[i]:cat_split_idx_oo_upper[i]], cat_oo.dec[cat_split_idx_oo_lower[i]:cat_split_idx_oo_upper[i]], catalog_weights=cat_oo.w[cat_split_idx_oo_lower[i]:cat_split_idx_oo_upper[i]], mask=mask, smth=smth_arcmin[i])
    
    # save the map if desired
    if write_maps:
        if not os.path.exists(os.path.join(save_path, "maps")):
            os.mkdir(os.path.join(save_path, "maps"))
        
        pkmap = os.path.join(save_path, "maps", "odmap_{:s}_{:d}_{:s}.fits".format(orient_mode, pct, binstr_orient))
        hp.write_map(pkmap, odmap, overwrite=True, dtype=np.float32)
    
    if orient_mode in ['asym_xy', 'asym_x', 'asym_y']:
        compute_xy_pol = True
    else:
        compute_xy_pol = False
    print("Getting orientations.")
    alpha, e, nu, x_pol, y_pol = sao.measure_orientation_QU(cat_so.ra[cat_split_idx_so_lower[i]:cat_split_idx_so_upper[i]], cat_so.dec[cat_split_idx_so_lower[i]:cat_split_idx_so_upper[i]], odmap, mode='density', compute_xy_pol=True, mask=mask)
    # save the quantities. There should be 1 alpha, e, nu etc measured per stacking-object.
    cat_so.alpha[cat_split_idx_so_lower[i]:cat_split_idx_so_upper[i]] = alpha
    cat_so.e[cat_split_idx_so_lower[i]:cat_split_idx_so_upper[i]] = e
    cat_so.nu[cat_split_idx_so_lower[i]:cat_split_idx_so_upper[i]] = nu
    cat_so.x_pol[cat_split_idx_so_lower[i]:cat_split_idx_so_upper[i]] = x_pol
    cat_so.y_pol[cat_split_idx_so_lower[i]:cat_split_idx_so_upper[i]] = y_pol

    end = time.time()
    print(f"Time elapsed for bin {i+1} out of {len(zlist_tot_oo)}: {end- zbin_start_time:.2f} seconds.")
    
tot_time = time.time() - zloop_begin
print(f"Total time for processing zbins was {tot_time:.0f} seconds, or {(tot_time/60.):.2f} minutes.")

# prep the dictionary for saving. 
# remove all rows with nans (e.g., some distances might not have been recorded)
print("Catalog has nans with zmin", cat_so.z[np.isnan(cat_so.alpha)].min(), "and zmax", cat_so.z[np.isnan(cat_so.alpha)].max())
cat_so.remove_nans()
# report final zmin, zmax of catalog
print("Final catalog zmin, zmax:", cat_so.z.min(), cat_so.z.max())

to_save = {key: np.asarray(getattr(cat_so, key)) for key in cat_so.__dict__.keys() if type(getattr(cat_so, key)) in [np.ndarray, list]}

print("All saved columns:", to_save.keys())
# make sure that the lengths of all arrays are the same
assert all(len(v) == len(cat_so.ra) for v in to_save.values()), "Not all arrays have the same length."
df = pd.DataFrame(to_save)

with open(save_file, 'w') as f:
    f.write(f"# Original SO catalog: {cat_so.pathInCatalog}\n") # point to original catalog for the records
    f.write(f"# Original OO catalog: {cat_oo.pathInCatalog}\n") # point to original catalog for the records
    f.write(f"# Original randoms catalog: {cat_ran.pathInCatalog}\n") # point to original catalog for the records
    f.write(f"# Smoothing scale used: {smth} Mpc\n")
    df.to_csv(f, index=False, header=True)
    
print(f"Final time: {time.time() - start:.2f} seconds.")