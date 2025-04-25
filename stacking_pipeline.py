import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
# from mpi4py import MPI
from pixell import enmap, utils
import catalog
from kmeans_radec import kmeans_sample
from stacking_functions import *
from postprocessing import Stack, radial_decompose_2D
import h5py
# start = time.time()
# get the MPI ingredients
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

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

# Describe the constraints on the input catalog
constraint_str = "desi_lrgs_nugt2_egtpt3_smth10"

orient = "sym" # options are "original", "random", "sym", "asym_x", "asym_y", "asym_xy"

cutout_size = 19.*u.Mpc # size of the cutout in comoving Mpc

nreg = 48 # number of chunks = number of processors to use

binsize = 10 # number of bins for statistics
########################################################



# add some function here to check if each map is an enmap, and convert from healpix to enmap otherwise

# read the orientation information
Cat = catalog.Catalog(name="standard", nameLong=constraint_str, pathInCatalog="/mnt/raid-cita/mlokken/data/desi/stacking_points/lrgs_zlim_elgclrgc_nu10gt2_e10gtpt3_o10_100pct.csv")

#### getting the region splits for errors ####
km = kmeans_sample(np.vstack((Cat.RA,Cat.DEC)).T, nreg, maxiter=100, tol=1.0e-5)
Cat.labels = km.labels # add labels to the Catalog object
#### getting the region splits for errors ####

# input the dictionary of maps to stack here
# format should be "type":"path/to/map.fits"
# maps should be enmaps
# eventually this can include the masks as well
maps = {
    "map1" : {
        "type" : "y",
        "path" : "/mnt/raid-cita/mlokken/data/act_ymaps/ilc_SZ_yy.fits",
        "shortname" : "ACT_y_fid"
    }
}

# if the unit of cutout_size is Mpc, then we need to convert it to degrees
if cutout_size.unit == u.Mpc:
    # based on the maximum redshift, set the max size for the cutouts
    zmax = np.max(Cat.Z)
    cutout_size_deg = 1/(cosmo.kpc_comoving_per_arcmin(zmax).to(u.Mpc/u.deg))*cutout_size
elif cutout_size.unit in [u.deg, u.arcmin, u.arcsec]:
    cutout_size_deg = cutout_size.to(u.deg)
else:
    raise ValueError("cutout_size must be in units of Mpc, degrees, arcminutes, or arcseconds")

cutout_resolution = (0.5*u.arcmin).to(u.deg)
print(f"will take thumbnails with size {cutout_size_deg:.2f} and resolution {cutout_resolution:.2f} deg.")

for m in maps:
    mappath = maps[m]["path"]
    sn = maps[m]['shortname']
    print(f"Reading map {mappath}")
    imap = enmap.read_map(maps[m]["path"])
    # Prepare to save to an HDF5 file
    with h5py.File(f'/mnt/scratch-lustre/mlokken/stacking/ACTxDESI/orient_by_desi_elgc+lrgc_100/stacks/enmap/{sn}_lrgs_zlim_elgclrgc_nu10gt2_e10gtpt3_o10_100pct', 'w') as f:
        # we will parallelize this part later. Will probably need to be done differently where separate regions are written to their own files and consolidated later.
        MapStack = None
        nobj_per_reg = []
        for n in range(nreg):
            print()
            # make the ChunkObj
            in_reg = Cat.labels==n
            ChunkObj = Chunk(Cat.RA[in_reg], Cat.DEC[in_reg], Cat.alpha[in_reg], Cat.x_asym[in_reg], Cat.y_asym[in_reg])
            # get the stack    
            stack_n = stackChunk(ChunkObj, imap, cutout_size_deg.value, cutout_resolution.value, orient=orient)
            # get the profiles
            r, Cr, Sr = radial_decompose_2D(stack_n, 5) # eventually I should start using Sr
            nobj_per_reg.append(ChunkObj.nObj)
            f.create_dataset(f'Cr_profiles_reg{n}', data=Cr)
            f.create_dataset(f'stack_reg{n}', data=stack_n)
        f.create_dataset('Nobj_per_region', data=np.asarray(nobj_per_reg))
    #     if MapStack is None: # on the first pass
    #         # initialize the object to hold the result
    #         MapStack = Stack(40, img_splits = [stack_n], profile_splits = Cr, Npks_splits=[ChunkObj.nObj])
    #         # add the next region to the MapStack object
    #         MapStack.img_splits = np.concatenate(MapStack.img_splits, stack_n, axis=0)
    #         MapStack.profile_splits.append(Cr)
    #         MapStack.Npks_splits.append(ChunkObj.nObj)
    #     print(f"Region {n} complete")
    # # get statistics
    # MapStack.bin_and_get_stats(10)
    # # save the MapStack object information to a file



# # list of indices for each of the nChunk chunks
# chunkIndices = [list(range(iChunk*chunkSize, (iChunk+1)*chunkSize)) for iChunk in range(nChunk)]
# # make sure not to miss the last few objects; add them to the last chunk
# chunkIndices[-1] = list(range((nChunk-1)*chunkSize, Cat.nObj))
'''
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
'''