import numpy as np
import os
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo, z_at_value
import time
import healpy as hp
import sys
import select_and_orient as sao
import yaml
h = (cosmo.H0/100.).value

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

nu_min = cfg["analysis"]["nu_min"]
nu_max = cfg["analysis"]["nu_max"]
e_min  = cfg["analysis"]["e_min"]
e_max  = cfg["analysis"]["e_max"]
# Mode for splitting the data along the line-of-sight: either custom_zlist, custom_dlist, auto_all, or auto_overlap (automatic bin sandwiching with predefined relative sizes)
los_split_mode = cfg["analysis"]["los_split_mode"]
# split if you want to only use some of the galaxy data to orient and other to stack
frac_use = cfg["analysis"]["fraction_input_data"]
# Smooth the maps by a Gaussian with this beam FWHM
smth     = cfg["analysis"]["smoothing_Mpc"]
orient_mode = cfg["analysis"]["orientation_mode"] # "original", "random", "sym", "asym_x", "asym_y", "asym_xy"
center_objects_label = cfg["analysis"]["center_objects_label"] # e.g., 'lrgc_dr1' for 'clustering' LRGs
orient_objects_label = cfg["analysis"]["orient_objects_label"] # e.g., 'elgc+lrgc_dr1' for 'clustering' LRGs + ELGs
nside = cfg["analysis"]["healpix_nside"]

use_mpi = cfg["mpi"]["use_mpi"]

##################################

if use_mpi:
    from mpi4py import MPI
    # get the MPI ingredients
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
else:
    rank = 0
    size = 1

smth_str = ("{:.1f}".format(smth)).replace('.','pt')

# define the width in comoving Mpc for stacking object bins
so_width = 5
# and for orientation object bins
oo_width = 20
pct = frac_use*100

# have rank 0 make the new directory, all others wait
if rank == 0:
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print(f"Created directory {save_path} for this run.")
if use_mpi and size > 1:
    comm.Barrier()  # wait for rank 0 to finish making the directory


if los_split_mode == 'custom_zlist':
    # enter the pre-defined redshift bins as the first argument
    zbins = cfg["analysis"]["z_bins"]
    dlist_tot, zlist_tot = sao.dlist(cosmo, zlist=zbins)

elif los_split_mode == 'custom_dlist':
    # enter the pre-defined comoving distance bins as the first argument
    dbins = cfg["analysis"]["chi_bins"]
    dlist_tot, zlist_tot = sao.dlist(cosmo, dlist=dbins)
    
elif los_split_mode == 'auto_overlap':
    minz = cfg["analysis"]["z_min"]+.005 # add small buffer
    maxz = cfg["analysis"]["z_max"]-.005 # add small buffer
    comoving_oo_narrowbin_start  = cosmo.comoving_distance(minz).to(u.Mpc)
    comoving_oo_narrowbin_0 = np.array([comoving_oo_narrowbin_start.value, (comoving_oo_narrowbin_start+oo_width*u.Mpc).value])
    nbins = int((cosmo.comoving_distance(maxz).to(u.Mpc).value - comoving_oo_narrowbin_start.value)/so_width)
    dbins = [comoving_oo_narrowbin_0 + i * so_width for i in range(nbins)] 
    print("Have you made sure to customize the width of the orientation and stacking slices? It is currently set to {:d} and {:d} Mpc with a mininum z of {:.2f} and maximum z of {:.2f}.".format(oo_width, so_width, minz, maxz))
    dlist_tot, zlist_tot = sao.dlist(cosmo, dlist=dbins)    

# save the zlist to a file
np.savetxt(os.path.join(save_path, "zlist.txt"), zlist_tot)

nside = 1024
npix  = hp.nside2npix(nside)
th, ph = hp.pixelfunc.pix2ang(nside, np.arange(npix))
cotth = np.cos(th)/np.sin(th)
# load the center objects data
ra_so,dec_so,z_so,w_so = sao.get_radecz(stack_catalog, return_weight=True)
# if different, load the orientation data
if orient_catalog is not None:
    ra_oo, dec_oo, z_oo, w_oo = sao.get_radecz(orient_catalog, return_weight=True)

# load the randoms data
ra_rand, dec_rand, z_rand, w_rand = sao.get_radecz(randoms_catalog, return_weight=True)

peakspath = save_path + "orient_by_{:s}_{:d}".format(orient_objects_label, pct)

orient_path = os.path.join(peakspath, "orientations")
if not os.path.exists(orient_path):
    os.mkdir(orient_path)
    
for i in range(len(zlist_tot)):
    
    start = time.time()
    dbin, zbin = dlist_tot[i], zlist_tot[i]
    oo_dlow,oo_dhi = dbin[0], dbin[1]
    oo_zlow,oo_zhi = zbin[0], zbin[1]
    bincent  = (oo_dlow+oo_dhi)/2.
    
    if los_split_mode=='auto_overlap' or los_split_mode=='custom_dlist':
        # find stacking objects only within plus/minus so_width/2 cMpc of the bin center
        so_dlow  = bincent-so_width/2.
        so_dhi   = bincent+so_width/2.
        so_zlow, so_zhi = z_at_value(cosmo.comoving_distance, so_dlow*u.Mpc).value, z_at_value(cosmo.comoving_distance, so_dhi*u.Mpc).value
        print("Finding stacking objects within {:.1f} cMpc of {:.0f} Mpc".format(so_width/2., bincent))
        print("In redshift space, this is between {:.2f} and {:.2f}.".format(so_zlow,so_zhi))
    elif los_split_mode=='auto_all' or los_split_mode=='custom_zlist':
        so_dlow, so_dhi = oo_dlow, oo_dhi
        so_zlow, so_zhi = oo_zlow, oo_zhi
        print("Finding stacking objects within the full bin.")
        
    
    if filenames_in_Mpc:
        # if so_dlow and so_dhi are round numbers, use them as ints
        if int(so_dlow)==so_dlow and int(so_dhi)==so_dhi:
            binstr_so   = "{:d}_{:d}Mpc".format(int(so_dlow), int(so_dhi))
        else:
            binstr_so   = ("{:.1f}_{:.1f}".format(so_dlow, so_dhi)).replace('.','pt')
    else: # save/expect redshifts in the filenames
        binstr_so   = ("{:.2f}_{:.2f}".format(so_zlow, so_zhi)).replace('.','pt')
        
    in_so_bin = (so_zlow<z_so)&(z_so<so_zhi)
    z_so_bin = z_so[in_so_bin]
    ra_so_bin = ra_so[in_so_bin]
    dec_so_bin = dec_so[in_so_bin]
    w_so_bin  = w_so[in_so_bin]
    
    print("Orienting by surrounding galaxies from {:.1f} to {:.1f} Mpc, {:.2f} to {:.2f}\n".format(oo_dlow,oo_dhi,oo_zlow,oo_zhi))
    if filenames_in_Mpc:
        if int(oo_dlow)==oo_dlow and int(oo_dhi)==oo_dhi:
            binstr_orient = "{:d}_{:d}Mpc".format(int(oo_dlow), int(oo_dhi))
        else:
            binstr_orient = ("{:.1f}_{:.1f}".format(oo_dlow, oo_dhi)).replace('.','pt')   
    else:
        binstr_orient = ("{:.2f}_{:.2f}".format(oo_zlow, oo_zhi)).replace('.','pt')
    

    # get the orientation objects in the orientation bin
    if orient_catalog is not None:
        in_oo_bin = (z_oo<oo_zhi)&(z_oo>oo_zlow)
        w_oo_bin = w_oo[in_oo_bin]
        ra_oo_bin = ra_oo[in_oo_bin]
        dec_oo_bin = dec_oo[in_oo_bin]
    else: # use the stacking objects, but in a wider z range than before
        in_oo_bin = (z_so<oo_zhi)&(z_so>oo_zlow)
        w_oo_bin = w_so[in_oo_bin]
        ra_oo_bin = ra_so[in_oo_bin]
        dec_oo_bin = dec_so[in_oo_bin]

    in_rand_bin = (z_rand<oo_zhi) & (z_rand>oo_zlow)
    w_rand_bin = w_rand[in_rand_bin]
    ra_rand_bin = ra_rand[in_rand_bin]
    dec_rand_bin = dec_rand[in_rand_bin]
    
    
    odmap = sao.delta_g(nside, ra_oo_bin, dec_oo_bin, ra_rand=ra_rand_bin, dec_rand=dec_rand_bin, catalog_weights=w_oo_bin, randoms_weights=w_rand_bin, smth=smth)
    
    # save the map
    if write_maps:
        pkmap = os.path.join(save_path, "odmap_{:s}_{:d}_{:s}.fits".format(orient_mode, pct, binstr_orient))
        hp.write_map(pkmap, odmap, overwrite=True, dtype=np.float32)
    
    start = time.time()
    ra_cut, dec_cut, z_cut, ca, sa = sao.measure_orientation(ra_so_bin, dec_so_bin, z_so_bin, pkmap, cotth, e_min=e_min, e_max=e_max, nu_min=nu_min, mode='density')
    
    end = time.time()
    print("Time elapsed for this bin: {:.2f} seconds".format(end-start))
    