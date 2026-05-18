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
minz = cfg["analysis"]["z_min"]
maxz = cfg["analysis"]["z_max"]
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
if frac_use != 1:
    pctstr = "_{:.0f}pct".format(pct)
else:
    pctstr = ""
if nu_min is not None or nu_max is not None or e_min is not None or e_max is not None:
    cutstr = '_cuts'
else:
    cutstr = ''
zstr = "{:.2f}_{:.2f}".format(minz, maxz).replace('.','pt')
# savename for file
save_file = os.path.join(save_path, f"{orient_objects_label}{pctstr}_{zstr}_{smth_str}Mpc{cutstr}_{orient_mode}.csv")
if os.path.exists(save_file):
    raise ValueError(f"Output file {save_file} already exists. Please change the save_path or delete the existing file to avoid overwriting.")
else:
    print("Will save output to", save_file)

if maskfile is not None:
    mask = hp.read_map(maskfile)
zlist_tot = None
ra_so_allranks = None
dec_so_allranks = None
z_so_allranks = None
w_so_allranks = None
ra_oo_allranks = None
dec_oo_allranks = None
z_oo_allranks = None
w_oo_allranks = None
ra_rand_allranks = None
dec_rand_allranks = None
z_rand_allranks = None
w_rand_allranks = None
counts_so = None
counts_oo = None
counts_rand = None
displays_oo = None
displays_so = None
displays_rand = None
zlist_chunks = None
dlist_chunks = None
# have rank 0 make the new directory and copy over config file, all others wait
if rank == 0:
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print(f"Created directory {save_path} for this run.")
    
    # copy the config file into the output directory, if not already there
    if not os.path.exists(os.path.join(save_path, os.path.basename(config_file_path))):
        shutil.copy(config_file_path, os.path.join(save_path, os.path.basename(config_file_path)))


    if los_split_mode == 'custom_zlist':
        # enter the pre-defined redshift bins as the first argument
        zbins = cfg["analysis"]["z_bins"]
        dlist_tot, zlist_tot = sao.dlist(cosmo, zlist=zbins)

    elif los_split_mode == 'custom_dlist':
        # enter the pre-defined comoving distance bins as the first argument
        dbins = cfg["analysis"]["chi_bins"]
        dlist_tot, zlist_tot = sao.dlist(cosmo, dlist=dbins)
        
    elif los_split_mode == 'auto_overlap':
        minz = minz+.005 # add small buffer
        maxz = maxz-.005 # add small buffer
        comoving_oo_narrowbin_start  = cosmo.comoving_distance(minz).to(u.Mpc)
        comoving_oo_narrowbin_0 = np.array([comoving_oo_narrowbin_start.value, (comoving_oo_narrowbin_start+oo_width*u.Mpc).value])
        nbins = int((cosmo.comoving_distance(maxz).to(u.Mpc).value - comoving_oo_narrowbin_start.value)/so_width)
        dbins = [comoving_oo_narrowbin_0 + i * so_width for i in range(nbins)] 
        print("Have you made sure to customize the width of the orientation and stacking slices? It is currently set to {:d} and {:d} Mpc with a mininum z of {:.2f} and maximum z of {:.2f}.".format(oo_width, so_width, minz, maxz))
        dlist_tot, zlist_tot = sao.dlist(cosmo, dlist=dbins)    

    # save the zlist to a file
    np.savetxt(os.path.join(save_path, "zlist.txt"), zlist_tot)

    # load the center objects data
    ra_so,dec_so,z_so,w_so = sao.get_radecz(stack_catalog, return_weight=True)
    # if different, load the orientation data
    if orient_catalog is not None:
        ra_oo, dec_oo, z_oo, w_oo = sao.get_radecz(orient_catalog, return_weight=True)

    if randoms_catalog is not None:
        # load the randoms data
        ra_rand, dec_rand, z_rand, w_rand = sao.get_radecz(randoms_catalog, return_weight=True)
        
    
    
    if size > 1:
        
        # divide all catalogs into z bins to be shared across the nodes
        # prepare the lists for sending
        ra_so_allranks = []
        dec_so_allranks = []
        z_so_allranks = []
        w_so_allranks = []
        if orient_catalog is not None:
            ra_oo_allranks = []
            dec_oo_allranks = []
            z_oo_allranks = []
            w_oo_allranks = []
        if randoms_catalog is not None:
            ra_rand_allranks = []
            dec_rand_allranks = []
            z_rand_allranks = []
            w_rand_allranks = []
            
        # split the zlist into 'size'-length chunks
        zlist_chunks = np.array_split(zlist_tot, size)
        dlist_chunks = np.array_split(dlist_tot, size)
        # for each chunk, take only the ra and dec data in that redshift and store to send to that rank
        for rank_i in range(size):
            z_min_i = zlist_chunks[rank_i][0][0] - 0.01 # add small buffer
            z_max_i = zlist_chunks[rank_i][-1][1] + 0.01 # add small buffer
            in_bin_i = (z_so>z_min_i) & (z_so<z_max_i)
            ra_so_allranks.append(ra_so[in_bin_i])
            dec_so_allranks.append(dec_so[in_bin_i])
            z_so_allranks.append(z_so[in_bin_i])
            w_so_allranks.append(w_so[in_bin_i])
            if orient_catalog is not None:
                in_bin_i = (z_oo>z_min_i) & (z_oo<z_max_i)
                ra_oo_allranks.append(ra_oo[in_bin_i])
                dec_oo_allranks.append(dec_oo[in_bin_i])
                z_oo_allranks.append(z_oo[in_bin_i])
                w_oo_allranks.append(w_oo[in_bin_i])
            if randoms_catalog is not None:
                in_bin_i = (z_rand>z_min_i) & (z_rand<z_max_i)
                ra_rand_allranks.append(ra_rand[in_bin_i])
                dec_rand_allranks.append(dec_rand[in_bin_i])
                z_rand_allranks.append(z_rand[in_bin_i])
                w_rand_allranks.append(w_rand[in_bin_i])

        counts_so = np.array([len(arr) for arr in ra_so_allranks])
        displays_so = np.insert(np.cumsum(counts_so), 0, 0)[0:-1] # the starting index for each rank's chunk in the concatenated array
        ra_so_allranks = np.concatenate(ra_so_allranks)
        dec_so_allranks = np.concatenate(dec_so_allranks)
        z_so_allranks = np.concatenate(z_so_allranks)
        w_so_allranks = np.concatenate(w_so_allranks)
        
        if orient_catalog is not None:
            counts_oo = np.array([len(arr) for arr in ra_oo_allranks])
            displays_oo = np.insert(np.cumsum(counts_oo), 0, 0)[0:-1]
            ra_oo_allranks = np.concatenate(ra_oo_allranks)
            dec_oo_allranks = np.concatenate(dec_oo_allranks)
            z_oo_allranks = np.concatenate(z_oo_allranks)
            w_oo_allranks = np.concatenate(w_oo_allranks)
            
        if randoms_catalog is not None:
            counts_rand = np.array([len(arr) for arr in ra_rand_allranks])
            displays_rand = np.insert(np.cumsum(counts_rand), 0, 0)[0:-1]
            ra_rand_allranks = np.concatenate(ra_rand_allranks)
            dec_rand_allranks = np.concatenate(dec_rand_allranks)
            z_rand_allranks = np.concatenate(z_rand_allranks)
            w_rand_allranks = np.concatenate(w_rand_allranks)  
        
if size>1 and rank>0:
    # comm.Barrier()  # wait for rank 0 to finish these tasks
    time_to_initiate = time.time()
    print(f"Time that rank {rank} waited to receive data: {time_to_initiate - start:.2f} seconds.")

if size>1:
    # Share metadata with all ranks
    counts_so = comm.bcast(counts_so, root=0)
    if orient_catalog is not None:
        counts_oo = comm.bcast(counts_oo, root=0)
    if randoms_catalog is not None:
        counts_rand = comm.bcast(counts_rand, root=0)
    zlist_chunks = comm.bcast(zlist_chunks, root=0)
    zlist_tot = zlist_chunks[rank]
    dlist_chunks = comm.bcast(dlist_chunks, root=0)
    dlist_tot = dlist_chunks[rank]
    # displs = comm.bcast(displs, root=0)
    # prepare to receive
    ra_so = np.empty(counts_so[rank], dtype=np.float64)
    # send the data with scatterv
    comm.Scatterv([ra_so_allranks, counts_so, displays_so, MPI.DOUBLE], ra_so, root=0)
    dec_so = np.empty(counts_so[rank], dtype=np.float64)
    comm.Scatterv([dec_so_allranks, counts_so, displays_so, MPI.DOUBLE], dec_so, root=0)
    z_so = np.empty(counts_so[rank], dtype=np.float64)
    comm.Scatterv([z_so_allranks, counts_so, displays_so, MPI.DOUBLE], z_so, root=0)
    w_so = np.empty(counts_so[rank], dtype=np.float64)
    comm.Scatterv([w_so_allranks, counts_so, displays_so, MPI.DOUBLE], w_so, root=0)
    if orient_catalog is not None:
        ra_oo = np.empty(counts_oo[rank], dtype=np.float64)
        comm.Scatterv([ra_oo_allranks, counts_oo, displays_oo, MPI.DOUBLE], ra_oo, root=0)
        dec_oo = np.empty(counts_oo[rank], dtype=np.float64)
        comm.Scatterv([dec_oo_allranks, counts_oo, displays_oo, MPI.DOUBLE], dec_oo, root=0)
        z_oo = np.empty(counts_oo[rank], dtype=np.float64)
        comm.Scatterv([z_oo_allranks, counts_oo, displays_oo, MPI.DOUBLE], z_oo, root=0)
        w_oo = np.empty(counts_oo[rank], dtype=np.float64)
        comm.Scatterv([w_oo_allranks, counts_oo, displays_oo, MPI.DOUBLE], w_oo, root=0)
    if randoms_catalog is not None:
        ra_rand = np.empty(counts_rand[rank], dtype=np.float64)
        comm.Scatterv([ra_rand_allranks, counts_rand, displays_rand, MPI.DOUBLE], ra_rand, root=0)
        dec_rand = np.empty(counts_rand[rank], dtype=np.float64)
        comm.Scatterv([dec_rand_allranks, counts_rand, displays_rand, MPI.DOUBLE], dec_rand, root=0)
        z_rand = np.empty(counts_rand[rank], dtype=np.float64)
        comm.Scatterv([z_rand_allranks, counts_rand, displays_rand, MPI.DOUBLE], z_rand, root=0)
        w_rand = np.empty(counts_rand[rank], dtype=np.float64)
        comm.Scatterv([w_rand_allranks, counts_rand, displays_rand, MPI.DOUBLE], w_rand, root=0)
    if maskfile is not None:
        mask = comm.bcast(mask, root=0)
    sending_time = time.time()
    print(f"Time that rank {rank} took to receive data: {sending_time - start:.2f} seconds.")
            
        
alpha_all = []
xpol_all = []
ypol_all = []
ca_all = []
sa_all = []
ra_all = []
dec_all = []
z_all = []

if zlist_tot is None:
    zlist_tot = np.loadtxt(os.path.join(save_path, "zlist.txt"))

# prepare the cotangent theta values for the healpix maps
npix  = hp.nside2npix(nside)
th, ph = hp.pixelfunc.pix2ang(nside, np.arange(npix))
cotth = np.cos(th)/np.sin(th)

#### This is where the main calculations happen ####
zloop_begin = time.time()
for i in range(len(zlist_tot)):
    zbin_start_time = time.time()
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

    if randoms_catalog is not None:
        in_rand_bin = (z_rand<oo_zhi) & (z_rand>oo_zlow)
        w_rand_bin = w_rand[in_rand_bin]
        ra_rand_bin = ra_rand[in_rand_bin]
        dec_rand_bin = dec_rand[in_rand_bin]
        
    z_mid = (oo_zlow + oo_zhi)/2.
    smth_arcmin = (cosmo.arcsec_per_kpc_comoving(z_mid) * (smth*u.Mpc)).to(u.arcmin).value
    if randoms_catalog is not None:
        odmap, mask = sao.delta_g(nside, ra_oo_bin, dec_oo_bin, ra_rand=ra_rand_bin, dec_rand=dec_rand_bin, catalog_weights=w_oo_bin, randoms_weights=w_rand_bin, smth=smth_arcmin)
    elif maskfile is not None:
        odmap = sao.delta_g(nside, ra_oo_bin, dec_oo_bin, catalog_weights=w_oo_bin, mask=mask, smth=smth_arcmin)
    
    
    # save the map
    if write_maps:
        if not os.path.exists(os.path.join(save_path, "maps")):
            os.mkdir(os.path.join(save_path, "maps"))
        
        pkmap = os.path.join(save_path, "maps", "odmap_{:s}_{:d}_{:s}.fits".format(orient_mode, pct, binstr_orient))
        hp.write_map(pkmap, odmap, overwrite=True, dtype=np.float32)
    
    
    if orient_mode in ['asym_xy', 'asym_x', 'asym_y']:
        return_xy_pol = True
    else:
        return_xy_pol = False
    print("Getting orientations.")
    alpha, x_pol, y_pol, ca, sa, final_cut = sao.measure_orientation(ra_so_bin, dec_so_bin, odmap, cotth, e_min=e_min, e_max=e_max, nu_min=nu_min, mode='density', return_xy_pol=return_xy_pol, mask=mask)
    
    alpha_all.extend(alpha)
    xpol_all.extend(x_pol)
    ypol_all.extend(y_pol)
    ca_all.extend(ca)
    sa_all.extend(sa)
    ra_all.extend(ra_so_bin[final_cut])
    dec_all.extend(dec_so_bin[final_cut])
    z_all.extend(z_so_bin[final_cut])
    
    end = time.time()
    print(f"Time elapsed for bin {i} out of {len(zlist_tot)} on rank {rank}: {end- zbin_start_time:.2f} seconds.")
    
tot_time = time.time() - zloop_begin
print(f"Total time for processing zbins was {tot_time:.0f} seconds, or {(tot_time/60.):.2f} minutes on rank {rank}.")

# all into arrays
alpha_all = np.asarray(alpha_all)
xpol_all = np.asarray(xpol_all)
ypol_all = np.asarray(ypol_all)
ca_all = np.asarray(ca_all)
sa_all = np.asarray(sa_all)
ra_all = np.asarray(ra_all)
dec_all = np.asarray(dec_all)
z_all = np.asarray(z_all)

    
if size>1:
    n_local = len(ra_all)
    # gather results
    counts = comm.gather(n_local, root=0)
    if rank == 0:
        counts = np.array(counts)
        displs = np.insert(np.cumsum(counts), 0, 0)[:-1]
        total_size  = np.sum(counts)
        recvbuf_ra = np.empty(total_size, dtype=np.float64)
        recvbuf_dec = np.empty(total_size, dtype=np.float64)
        recvbuf_z = np.empty(total_size, dtype=np.float64)
        recvbuf_alpha = np.empty(total_size, dtype=np.float64)
        recvbuf_xpol = np.empty(total_size, dtype=np.int32)
        recvbuf_ypol = np.empty(total_size, dtype=np.int32)
    else:
        recvbuf_ra = None
        recvbuf_dec = None
        recvbuf_z = None
        recvbuf_alpha = None
        recvbuf_xpol = None
        recvbuf_ypol = None
        counts = None
        displs = None


    comm.Gatherv(np.asarray(ra_all), [recvbuf_ra, counts, displs, MPI.DOUBLE], root=0)
    comm.Gatherv(np.asarray(dec_all), [recvbuf_dec, counts, displs, MPI.DOUBLE], root=0)
    comm.Gatherv(np.asarray(z_all), [recvbuf_z, counts, displs, MPI.DOUBLE], root=0)
    comm.Gatherv(np.asarray(alpha_all), [recvbuf_alpha, counts, displs, MPI.DOUBLE], root=0)    
    comm.Gatherv(np.asarray(xpol_all), [recvbuf_xpol, counts, displs, MPI.INT], root=0)
    comm.Gatherv(np.asarray(ypol_all), [recvbuf_ypol, counts, displs, MPI.INT], root=0)

    ra_all = recvbuf_ra
    dec_all = recvbuf_dec
    z_all = recvbuf_z
    alpha_all = recvbuf_alpha
    xpol_all = recvbuf_xpol
    ypol_all = recvbuf_ypol
    

# only rank 0 writes to file
if rank==0:
    df = pd.DataFrame({'RA':ra_all, 'DEC':dec_all, 'Z':z_all, 'alpha':alpha_all, 'x_asym':xpol_all, 'y_asym':ypol_all, 'config':os.path.basename(config_file_path).replace('.yaml','')})
    df.to_csv(save_file, index=False)
    
print(f"Final time: {time.time() - start:.2f} seconds.")