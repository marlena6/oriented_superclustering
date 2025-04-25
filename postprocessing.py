import numpy as np
class Stack:
    # an object to be loaded in from a file of an errors run
    # Not using any Astropy Quantities in this class because they cause bugs
    def __init__(self, rad_in_Mpc, avg_img=None, avg_profiles=None, img_splits=None, profile_splits=None, Npks_splits=None, Npks_tot=None):
        # Img is an array of shape (img_side_len, img_side_len)
        # avg_profiles is an array of shape (m_max, img_side_len//2)
        # Img_splits is an array of shape (N_splits, img_side_len, img_side_len)
        # Profile_splits is a list of length m_max, each element of list is array of shape (N_splits, img_side_len//2)
        # Npks_splits is an array of shape (N_splits,)
        # Rad_in_Mpc is the radius of the stack image in Mpc
        
        # begin with some checks
        if avg_img is None and img_splits is None:
            print("Must provide either img or img_splits.")
            return
        if avg_img is not None and type(avg_img) != np.ndarray:
            print("img must be a numpy array.")
            return
        if avg_profiles is not None and type(avg_profiles) not in [np.ndarray, list]:
            print("profiles must be a numpy array or list.")
            return
        # Convert some lists to arrays if necessary
        if img_splits is not None and type(img_splits) not in [np.ndarray, list]:
            print("img_splits must be a numpy array or list.")
            return
        if profile_splits is not None and type(profile_splits) not in [np.ndarray, list]:
            print("profile_splits must be a numpy array or list.")
            return
        if type(img_splits)  == list:
            img_splits = np.asarray(img_splits)
        if type(Npks_splits) == list:
            Npks_splits = np.asarray(Npks_splits)
        
        self.__has_splits__ = False
        if img_splits is not None:
            self.__has_splits__ = True
        
        self.avg_img         = avg_img # stack image
        self.img_splits      = img_splits # stack images in splits. The weighted average of these should be the full stack image
        self.profile_splits  = profile_splits # unbinned multipole profiles in splits
        self.rad_in_Mpc      = rad_in_Mpc # radius of the stack image in Mpc
        self.Npks_splits     = Npks_splits # number of peaks in each split
        if self.Npks_splits is not None:
            self.Npks_tot    = np.sum(self.Npks_splits)
        else:
            self.Npks_tot    = Npks_tot
        self.avg_profiles    = avg_profiles # list of the unbinned average profiles for each multipole moment m. Length m_max, each element shape (n_bins,)
        
        if self.__has_splits__:
            self.Nsamples = len(img_splits) # number of samples
            self.split_wgts = self.Npks_splits / np.average(self.Npks_splits)
            if profile_splits is not None:
                self.mmax = len(profile_splits) # maximum multipole moment
        if self.avg_profiles is None and self.profile_splits is not None: # if avg_profiles is not provided, calculate it from the splits
            self.avg_profiles = []
            for m,profsplits in enumerate(self.profile_splits):
                self.avg_profiles.append(np.average(profsplits, axis=0, weights=self.split_wgts))
        if self.avg_img is None:
            self.avg_img = np.average(self.img_splits, axis=0, weights=self.split_wgts)

        self.r = np.arange(1, self.avg_img.shape[0]//2) * self.rad_in_Mpc / (self.avg_img.shape[0]//2)  # unbinned radius variable in Mpc
        # if self.r not equal to profile_splits.shape[2], print warning
        if self.profile_splits is not None:
            if len(self.r) != self.avg_profiles[0].shape[0]:
                print("Warning: r and profile_splits are different lengths.")
        # Initialize optional attributes to None
        self.covmat_full     = []
        self.cormat_full     = []
        self.errors_full     = []
        self.covmat_binned   = []
        self.cormat_binned   = []
        self.errors_binned   = [] # errors on the binned profile
        

        
    def set_split_wgts(self, additional_weights=None):
        # optionally replace split_wgts
        # if additional_weights is None, weights depend only on number of peaks in each split
        if additional_weights is None:
            additional_weights = np.ones(self.Nsamples)
        self.split_wgts = self.Npks_splits / np.average(self.Npks_splits) * additional_weights
    def set_average_profiles(self): # Option to call this by hand, to reset the profile, if the weights have changed
        self.avg_profiles = []
        for m,profsplits in enumerate(self.profile_splits):
            self.avg_profiles.append(np.average(profsplits, axis=0, weights=self.split_wgts))
    def set_avg_profiles_binned(self, binsize):
        # a list of the average binned profiles for each multipole moment m. Length m_max, each element shape (n_bins,)
        self.avg_profiles_binned = []
        for m,avgprof in enumerate(self.avg_profiles):
            binned_prof, binned_r = bin_profile(self.r, avgprof, self.rad_in_Mpc, binsize)
            self.avg_profiles_binned.append(np.asarray(binned_prof))
        self.r_binned = np.asarray(binned_r) # set binned r as whatever the last binned r was. These should all be the same.

    def set_custom_bin_m_avg(self, m, custom_bins):
        # rebin the mth multipole moment of the profiles
        custom_bins = np.asarray(custom_bins)
        custom_profile_m = [np.average(self.avg_profiles[m][custom_bins[i]:custom_bins[i+1]]) for i,bin in enumerate(custom_bins[:-1])]
        self.avg_profiles_binned[m] = np.asarray(custom_profile_m)
        self.r_binned = xcenters = (custom_bins[:-1] + custom_bins[1:]) / 2 * self.rad_in_Mpc / (self.avg_img.shape[0]//2)
    def set_profile_splits_binned(self, binsize): # bin the profile of each split
        if not self.__has_splits__:
            print("No splits to bin.")
            return
        else:
            profile_splits_binned = []
            for m,profsplits in enumerate(self.profile_splits):
                profile_splits_binned_m = []
                for split in profsplits:
                    binned_prof, binned_r = bin_profile(self.r, split, self.rad_in_Mpc, binsize)
                    profile_splits_binned_m.append(np.asarray(binned_prof))
                profile_splits_binned.append(np.asarray(profile_splits_binned_m))
            self.profile_splits_binned = profile_splits_binned # list with len(m_max), each element shape (n_splits, n_bins)
            # not making into array because each element may have different shape after reassignment; see set_custom_bin_m
    def set_custom_bin_m_splits(self, m, custom_bins):
        # rebin the mth multipole moment of the profiles
        if not self.__has_splits__:
            print("No splits to bin.")
            return
        else:
            custom_profile_m = [np.average(self.profile_splits[m][:,custom_bins[i]:custom_bins[i+1]], axis=1) for i,bin in enumerate(custom_bins[:-1])]
            self.profile_splits_binned[m] = np.asarray(custom_profile_m).transpose()
    def set_covariance_full(self):
        # set the covariance matrix for the full profile
        if not self.__has_splits__:
            print("No splits to bin.")
            return
        else:
            for m,profsplits in enumerate(self.profile_splits):
                covmat, cormat = covariances(profsplits, self.split_wgts, self.Nsamples)
                self.covmat_full.append(covmat)
                self.cormat_full.append(cormat)
                self.errors_full.append(np.sqrt(np.diag(covmat)))
    def set_covariance_binned(self):
        # set the covariance matrix for the binned profile
        # reset in case already set
        self.covmat_binned = []
        self.cormat_binned = []
        self.errors_binned = []
        if not self.__has_splits__:
            print("No splits to bin.")
            return
        else:
            for m, profsplits in enumerate(self.profile_splits_binned):
                covmat, cormat = covariances(profsplits, self.split_wgts, self.Nsamples)
                self.covmat_binned.append(covmat)
                self.cormat_binned.append(cormat)
                self.errors_binned.append(np.sqrt(np.diag(covmat)))
    def bin_and_get_stats(self, binsize):
        self.set_profile_splits_binned(binsize)
        self.set_avg_profiles_binned(binsize)
        self.set_covariance_full()
        self.set_covariance_binned()
        
def bin_profile(r, Cr_m, rad_in_mpc, binsize): # binsize in Mpc
    # r is in Mpc
    
    npix    = len(Cr_m)
    mpc_per_pix = rad_in_mpc / (npix-1)
    pix_per_bin = int(np.floor((binsize / mpc_per_pix)))
    binned_prof = []
    binned_r    = []
    for i in range(0,npix-pix_per_bin,pix_per_bin):
        binned_prof.append(np.mean(Cr_m[i:i+pix_per_bin]))
        binned_r.append(np.mean(r[i:i+pix_per_bin]))
    #r = [i + 0.5 for i in range(len(binned_prof))]
    #step_in_mpc = rad_in_mpc/len(r)
    return binned_prof, binned_r

def covariances(y_list,weights,nreg):
    y_array = np.asarray(y_list)
    covmat = np.cov(y_array.T, aweights=weights)/nreg
    correl_mat = cormat(covmat)
    return(covmat, correl_mat)

def cormat(covmat):
    cormat = np.zeros(covmat.shape)
    for i in range(covmat.shape[0]):
        for j in range(covmat.shape[1]):
            corcoeff = covmat[i,j]/np.sqrt(covmat[i,i]*covmat[j,j])
            cormat[i,j] = corcoeff
    return(cormat)

def radial_decompose_2D(f, mmax):
        # f is numpy image array                                                                                                                                                                                  
        # mmax is maximum m for decomposition (maximally 10)                                                                                                                                                     
        n = int(f.shape[0] // 2)
        nsteps = n * 20
        dtheta = 2*np.pi/nsteps
        Cr = np.zeros((n-1, mmax))
        Sr = np.zeros((n-1, mmax))

        for i in range(1, n):
                r = float(i)
                for j in range(nsteps):
                        # print(j)                                                                                                                                                                                
                        theta = dtheta * j
                        # print(theta)                                                                                                                                                                            
                        rx    = r*np.cos(theta)
                        ry    = r*np.sin(theta)
                        ix    = int(min(np.floor(rx), n-1))
                        iy    = int(min(np.floor(ry), n-1))
                        rx    = rx - ix
                        ry    = ry - iy
                        ix    = ix + n # different from Fortran COOP version -- indexing middle of array                                                                                                          
                        iy    = iy + n
                        fv    = (1-rx)*(f[iy, ix]*(1-ry) + f[iy+1, ix]*ry) + rx * ( f[iy, ix+1]*(1-ry) + f[iy+1, ix+1]*ry)
                        Cr[i-1,0] += fv
                        for m in range(1, mmax):
                                Cr[i-1,m] = Cr[i-1,m] + fv * np.cos(m*theta)
                                Sr[i-1,m] = Sr[i-1,m] + fv * np.sin(m*theta)
                
        Cr[0:n-1,0] = Cr[0:n-1, 0]/nsteps
        Cr[0:n-1,1:mmax] =  Cr[0:n-1,1:mmax] * (2./nsteps)
        Sr[0:n-1,1:mmax] =  Sr[0:n-1,1:mmax] * (2./nsteps)
        return(np.arange(1,n), Cr, Sr)