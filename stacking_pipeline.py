import numpy as np
import error_analysis_funcs as ef
import os
from astropy.cosmology import Planck18 as cosmo, z_at_value
import astropy.units as u
import subprocess
import coop_post_processing as cpp
import coop_setup_funcs as csf
from astropy.io import fits
import matplotlib.pyplot as plt
from mpi4py import MPI
import pandas as pd
from pixell import enmap
import catalog

# start = time.time()
# get the MPI ingredients
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

h = (cosmo.H0/100.).value

# ALL CHOICES COME BELOW
########################################################
# mode is which data set, uncomment one of the following
# mode  = 'Buzzard'
# mode  = 'Cardinal'
# mode = 'ACTxDES'
mode = 'ACTxDESI'
# mode = 'Websky'
# mode = 'GRF'

errors = False # if true, split regions to get error estimates

# Smooth the maps by a Gaussian with this beam FWHM
constraint_str = "desi_lrgs_nugt2_egtpt3_smth10"
# Also should come with a header that describes all the information

orient = "sym" # options are "original", "random", "sym", "asym_x", "asym_y", "asym_xy"
########################################################


#### section for getting the region splits for errors, to do later
### if errors:
###     nreg = 48
###     # read the ra and dec from input
    # then do the labels split
################

# input the dictionary of maps to stack here
# format should be "type":"path/to/map.fits"
# maps should be enmaps
maps = {
    "map1" : {
        "type" : "y",
        "path" : "/mnt/raid-cita/mlokken/data/act_ymaps/ilc_SZ_yy.fits",
        "shortname" : "ACT_y_fid"
    }
}


# add some function here to check if each map is an enmap, and convert from healpix to enmap otherwise

# read the orientation information
Cat = catalog.Catalog(name="standard", nameLong=constraint_str, pathInCatalog="/mnt/raid-cita/mlokken/data/desi/stacking_points/lrgs_zlim_elglrg_nu10gt2_e10gtpt3_o10.csv")


nruns_local = len(dlist_tot) // size
if rank == size-1:
    extras = len(dlist_tot) % size
else:
    extras = 0
# times = []
for n in range(nruns_local):
    i = rank*nruns_local+n
    print("Rank {:d}, bin {:d}".format(rank, i+1))
    dlow, dhi = dlist_tot[i][0], dlist_tot[i][1]
    zlow, zhi = zlist_tot[i][0], zlist_tot[i][1]
    bincent  = (dlow+dhi)/2.
    
    if los_split_mode=='auto_overlap' or los_split_mode=='custom_dlist':
        # find stacking objects only within plus/minus so_width/2 cMpc of the bin center
        so_dlow  = bincent-so_width/2.
        so_dhi   = bincent+so_width/2.
        so_zlow, so_zhi = z_at_value(cosmo.comoving_distance, so_dlow*u.Mpc).value, z_at_value(cosmo.comoving_distance, so_dhi*u.Mpc).value
        print("Finding stacking objects within {:.1f} cMpc of {:.0f} Mpc".format(so_width/2., bincent))
        print("In redshift space, this is between {:.2f} and {:.2f}.".format(so_zlow,so_zhi))
    elif los_split_mode=='auto_all' or los_split_mode=='custom_zlist':
        so_dlow, so_dhi = dlow, dhi
        so_zlow, so_zhi = zlow, zhi
        print("Finding stacking objects within the full bin.")
        
    slice_included=False # set this to false, it will change to true if this slice is encompassed by a galaxy number density map
    if filenames_in_Mpc:
        # if so_dlow and so_dhi are round numbers, use them as ints
        if int(so_dlow)==so_dlow and int(so_dhi)==so_dhi:
            binstr_so   = "{:d}_{:d}Mpc".format(int(so_dlow), int(so_dhi))
        else:
            binstr_so   = ("{:.1f}_{:.1f}".format(so_dlow, so_dhi)).replace('.','pt')
    else: # save/expect redshifts in the filenames
        binstr_so   = ("{:.2f}_{:.2f}".format(so_zlow, so_zhi)).replace('.','pt')
    
    if filenames_in_Mpc:
        if int(dlow)==dlow and int(dhi)==dhi:
            binstr_orient = "{:d}_{:d}Mpc".format(int(dlow), int(dhi))
        else:
            binstr_orient = ("{:.1f}_{:.1f}".format(dlow, dhi)).replace('.','pt')   
    else:
        binstr_orient = ("{:.2f}_{:.2f}".format(zlow, zhi)).replace('.','pt')
    # make the ini files
    if orient:
        orientstr="orient{:s}_{:d}pct_{:s}_{:s}".format(style, pct, orient_mode, binstr_orient)
    else:
        orientstr="randrot"
    print("NOW PROCESSING REDSHIFT BIN", binstr_orient)
    if mode=='ACTxDES':
        inifile_root = "redmapper_{:s}_{:s}_{:s}{:s}_{:s}".format(cutstr, binstr_so, pt_selection_str, smth_str, orientstr)
    elif mode=='ACTxDESI':
        inifile_root = "lrg_{:s}_{:s}_{:s}{:s}_{:s}".format(cutstr, binstr_so, pt_selection_str, smth_str, orientstr)
    pksfile = os.path.join(outpath+"orient_by_{:s}_{:d}/".format(orient_mode, pct), inifile_root+"_pks.fits")
    if errors:
        labels       = np.loadtxt("/home/mlokken/oriented_stacking/general_code/labels_{:d}_regions_{:s}_{:s}.txt".format(nreg,mode,cutstr))
        # cl_zlow      = z_at_value(cosmo.comoving_distance, cl_dlow*u.Mpc).value
        # cl_zhi       = z_at_value(cosmo.comoving_distance, cl_dhi*u.Mpc).value
        # cl_inbin     = (cl_zlow<z_cl)&(z_cl<cl_zhi) # just need this for the boolean array to subselect from labels list
        # labels_inbin = labels[cl_inbin]
        #cm = plt.get_cmap('gist_rainbow')\
        for reg in range(nreg):
            # start = time.time()
            regpath = os.path.join(stkpath,"{:d}".format(reg))
            if not os.path.exists(regpath):
                print("Making {:s}".format(regpath))
                os.mkdir(regpath)
            with fits.open(pksfile) as pks:
                pkdata = pks[0].data
                ncols = pkdata.shape[1]
                colnames = ["id","theta","phi","rot_angle","x_up", "y_up"]
                pd_pkdata = pd.DataFrame(data=pkdata.byteswap().newbyteorder(), columns=colnames[:ncols]) # make the peak info a dataframe so we can merge
                pd_labels = pd.DataFrame(data=labels, columns=["id","reg_label"])
                pks_w_labels = pd.merge(pd_pkdata, pd_labels, how='left', on="id")
                # use the labels to split the data
                in_reg = pks_w_labels["reg_label"] == reg
                pkdata_new = pkdata[in_reg]
                print("Peak data in region {:d} and this distance bin:".format(reg), len(pkdata_new))
                if len(pkdata_new)>0:
                    pksfile_reg = os.path.join(regpath, inifile_root+"_reg{:d}".format(reg)+"_pks.fits")
                    pks[0].data = pkdata_new
                    # save a new pks fits file with only pkdata in region, and run stack on that
                    pks.writeto(pksfile_reg, overwrite=True)
                    # for plotting
                    # angle, ra, dec = cpp.peakinfo_radec(pksfile)
                    #plt.scatter(ra[in_reg], dec[in_reg], c=cm(reg/48))
                    #if reg==47:
                    #    plt.show()
                    # make the ini files
            if len(pkdata_new)>0:
                if stack_kappa:
                    k_inifile_root = kmode+"_"+inifile_root+"_reg{:d}".format(reg)
                y_inifile_root = ymode + "_"+inifile_root+"_reg{:d}".format(reg)
                m_inifile_root = "DES_mask_"+inifile_root+"_reg{:d}".format(reg)
                if stack_galaxies:
                    zbins_ndmaps = [[0.2,0.36],[0.36,0.53],[0.53,0.72],[0.72,0.94]]
                    for zbin in zbins_ndmaps:
                        if (z_at_value(cosmo.comoving_distance, so_dlow*u.Mpc)>=zbin[0]) & (z_at_value(cosmo.comoving_distance, so_dhi*u.Mpc)<zbin[1]):
                            zlow_str = ("{:.2f}".format(zbin[0])).replace('.', 'pt')
                            zhi_str  = ("{:.2f}".format(zbin[1])).replace('.', 'pt')
                            map_to_stack = pkmap_path+"ndmap_25_z_{:s}_{:s}.fits".format(zlow_str, zhi_str)
                            g_inifile_root = "{:s}_maglim_z_{:s}_{:s}_".format(gmode, zlow_str,zhi_str)+inifile_root+"_reg{:d}".format(reg)
                            slice_included = True
                    if slice_included:
                        if not os.path.exists(os.path.join(regpath, g_inifile_root+"_stk.fits")): # only try to do a g stack if there's a number density map that encompasses this slice 
                            gstk_ini = ef.make_stk_ini_file(pksfile_reg, map_to_stack, standard_stk_file_errs, regpath, g_inifile_root,[dlow,dhi], stk_mask=gmask, rad_Mpc=40)
                            print("Rank {:d} running Stack on {:s}".format(rank, gstk_ini))
                            subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",gstk_ini])
                            # remove extraneous files
                            os.remove(os.path.join(regpath, g_inifile_root+"_stk.txt"))
                            os.remove(os.path.join(regpath, g_inifile_root+"_stk.patch"))
                        elif os.path.exists(os.path.join(regpath, g_inifile_root+"_stk.fits")):
                            print("Galaxy map already stacked. Moving on.")
                    else:
                        print("WARNING: this slice does not fall within a number density map.")
                if stack_kappa and (not os.path.exists(os.path.join(regpath, k_inifile_root+"_stk.fits"))):
                    kstk_ini = ef.make_stk_ini_file(pksfile_reg, kappamap, standard_stk_file_errs, regpath, k_inifile_root, [dlow,dhi], stk_mask=kappamask, rad_Mpc=40)
                    print("Rank {:d} running Stack on {:s}".format(rank, kstk_ini))
                    subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",kstk_ini])
                    # remove extraneous files                                                                                                                                     
                    os.remove(os.path.join(regpath, k_inifile_root+"_stk.txt"))
                    os.remove(os.path.join(regpath, k_inifile_root+"_stk.patch"))
                elif stack_kappa and os.path.exists(os.path.join(regpath, k_inifile_root+"_stk.fits")):
                    print("Kappa map already stacked. Moving on.")
                if stack_y & (not os.path.exists(os.path.join(regpath, y_inifile_root+"_stk.fits"))):
                    stk_ini = ef.make_stk_ini_file(pksfile_reg, ymap, standard_stk_file_errs, regpath, y_inifile_root, [dlow,dhi], stk_mask=ymask, rad_Mpc=40)
                    print("Rank {:d} running Stack on {:s}".format(rank,stk_ini))
                    subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",stk_ini])
                    # remove extraneous files                                                                                                                                     
                    os.remove(os.path.join(regpath, y_inifile_root+"_stk.txt"))
                    os.remove(os.path.join(regpath, y_inifile_root+"_stk.patch"))
                elif stack_y and os.path.exists(os.path.join(regpath, y_inifile_root+"_stk.fits")):
                    print("Y map already stacked. Moving on.")
                if stack_mask and (not os.path.exists(os.path.join(regpath, m_inifile_root+"_stk.fits"))):
                    mstk_ini = ef.make_stk_ini_file(pksfile_reg, gmask, standard_stk_file_errs, regpath, m_inifile_root, [dlow,dhi], stk_mask=gmask, rad_Mpc=40)
                    print("Rank {:d} running Stack on {:s}".format(rank, mstk_ini))
                    subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",mstk_ini])
                    # remove extraneous files                                                                                                                                     
                    os.remove(os.path.join(regpath, m_inifile_root+"_stk.txt"))
                    os.remove(os.path.join(regpath, m_inifile_root+"_stk.patch"))
                elif stack_mask and os.path.exists(os.path.join(regpath, m_inifile_root+"_stk.fits")):
                    print("Mask already stacked. Moving on.")
                # end = time.time()
                # times.append(end-start)
    else:
        if stack_kappa:
            k_inifile_root = kmode+inifile_root
        y_inifile_root = ymode+"_"+inifile_root
        # start = time.time()
        if stack_galaxies:
            zbins_ndmaps = [[0.2,0.36],[0.36,0.53],[0.53,0.72],[0.72,0.94]]
            for zbin in zbins_ndmaps:
                print(zbin, so_dlow, so_dhi)
                if (z_at_value(cosmo.comoving_distance, so_dlow*u.Mpc)>zbin[0]) & (z_at_value(cosmo.comoving_distance, so_dhi*u.Mpc)<zbin[1]):
                    zlow_str = ("{:.2f}".format(zbin[0])).replace('.', 'pt')
                    zhi_str  = ("{:.2f}".format(zbin[1])).replace('.', 'pt')
                    map_to_stack = pkmap_path+"ndmap_25_z_{:s}_{:s}.fits".format(zlow_str, zhi_str)
                    g_inifile_root = "{:s}_maglim_z_{:s}_{:s}_".format(gmode,zlow_str,zhi_str)+inifile_root
            if not os.path.exists(os.path.join(stkpath, g_inifile_root+hankelstr)):
                gstk_ini = ef.make_stk_ini_file(pksfile, map_to_stack, standard_stk_file, stkpath, g_inifile_root,[dlow,dhi], stk_mask=gmask, rad_Mpc=40)
                print("Rank {:d} running Stack on {:s}".format(rank, gstk_ini))
                subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",gstk_ini])
                # remove extraneous files                                                                                                                                     
                os.remove(os.path.join(stkpath, g_inifile_root+"_stk.txt"))
                os.remove(os.path.join(stkpath, g_inifile_root+"_stk.patch"))
        if stack_kappa and (not os.path.exists(os.path.join(stkpath, k_inifile_root+hankelstr))):
            kstk_ini = ef.make_stk_ini_file(pksfile, kappamap, standard_stk_file, stkpath, k_inifile_root, [dlow,dhi], stk_mask=kappamask, rad_Mpc=40)
            print("Rank {:d} running Stack on {:s}".format(rank, kstk_ini))
            subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",kstk_ini])
            # remove extraneous files                                                                                                                                     
            os.remove(os.path.join(stkpath, k_inifile_root+"_stk.txt"))
            os.remove(os.path.join(stkpath, k_inifile_root+"_stk.patch"))
        if stack_y & (not os.path.exists(os.path.join(stkpath, y_inifile_root+hankelstr))):
            stk_ini = ef.make_stk_ini_file(pksfile, ymap, standard_stk_file, stkpath, y_inifile_root, [dlow,dhi], stk_mask=ymask, rad_Mpc=40)
            print("Rank {:d} running Stack on {:s}".format(rank,stk_ini))
            subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",stk_ini])
            # remove extraneous files                                                                                                                                     
            os.remove(os.path.join(stkpath, y_inifile_root+"_stk.txt"))
            os.remove(os.path.join(stkpath, y_inifile_root+"_stk.patch"))
        # end = time.time()
        # times.append(end-start)
# print("Rank, time list :", rank, times)
# print("Rank, average time per loop:", rank, np.average(times))
