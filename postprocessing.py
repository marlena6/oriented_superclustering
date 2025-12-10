import numpy as np
import stack_statistics as ss


class Stack:
    # an object to be loaded in from a file of an errors run
    # Not using any Astropy Quantities in this class because they cause bugs
    def __init__(
        self,
        radius,
        avg_img=None,
        Cr_avg_profiles=None,
        Sr_avg_profiles=None,
        img_splits=None,
        Cr_profile_splits=None,
        Sr_profile_splits=None,
        Npks_splits=None,
        Npks_tot=None,
        r=None,
    ):
        # Img is an array of shape (img_side_len, img_side_len)
        # avg_profiles is an array of shape (m_max, img_side_len//2)
        # Img_splits is an array of shape (N_splits, img_side_len, img_side_len)
        # Profile_splits is a list of length m_max, each element of list is array of shape (N_splits, img_side_len//2)
        # Npks_splits is an array of shape (N_splits,)
        # Radius is the radius of the stack image in whatever units

        # begin with some checks
        if avg_img is None and img_splits is None:
            print("Must provide either img or img_splits.")
            return
        if avg_img is not None and type(avg_img) is not np.ndarray:
            print("img must be a numpy array.")
            return
        if Cr_avg_profiles is not None and type(Cr_avg_profiles) not in [
            np.ndarray,
            list,
        ]:
            print("profiles must be a numpy array or list.")
            return
        # Convert some lists to arrays if necessary
        if img_splits is not None and type(img_splits) not in [np.ndarray, list]:
            print("img_splits must be a numpy array or list.")
            return
        if Cr_profile_splits is not None and type(Cr_profile_splits) not in [
            np.ndarray,
            list,
        ]:
            print("profile_splits must be a numpy array or list.")
            return
        if type(img_splits) is list:
            img_splits = np.asarray(img_splits)
        if type(Npks_splits) is list:
            Npks_splits = np.asarray(Npks_splits)

        self.__has_splits__ = False
        if img_splits is not None:
            self.__has_splits__ = True

        self.avg_img = avg_img  # stack image
        self.img_splits = img_splits  # stack images in splits. The weighted average of these should be the full stack image
        self.Cr_profile_splits = (
            Cr_profile_splits  # unbinned multipole profiles in splits
        )
        self.Sr_profile_splits = (
            Sr_profile_splits  # unbinned multipole profiles in splits
        )
        self.radius = radius  # radius of the stack image
        self.Npks_splits = Npks_splits  # number of peaks in each split
        if self.Npks_splits is not None:
            self.Npks_tot = np.sum(self.Npks_splits)
        else:
            self.Npks_tot = Npks_tot
        self.Cr_avg_profiles = Cr_avg_profiles  # list of the unbinned average profiles for each multipole moment m. Length m_max, each element shape (n_bins,)
        self.Sr_avg_profiles = Sr_avg_profiles

        if self.__has_splits__:
            self.Nsamples = len(img_splits)  # number of samples
            self.split_wgts = self.Npks_splits / np.average(self.Npks_splits)
            print("Number of splits", len(self.split_wgts))
            if Cr_profile_splits is not None:
                self.mmax = len(Cr_profile_splits)  # maximum multipole moment
        if (
            self.Cr_avg_profiles is None and self.Cr_profile_splits is not None
        ):  # if avg_profiles is not provided, calculate it from the splits
            self.Cr_avg_profiles = []
            for m, profsplits in enumerate(self.Cr_profile_splits):
                self.Cr_avg_profiles.append(
                    np.average(profsplits, axis=0, weights=self.split_wgts)
                )
        if (
            self.Sr_avg_profiles is None and self.Sr_profile_splits is not None
        ):  # if avg_profiles is not provided, calculate it from the splits
            self.Sr_avg_profiles = []
            for m, profsplits in enumerate(self.Sr_profile_splits):
                self.Sr_avg_profiles.append(
                    np.average(profsplits, axis=0, weights=self.split_wgts)
                )
        if self.avg_img is None:
            self.avg_img = np.average(self.img_splits, axis=0, weights=self.split_wgts)
        if r is not None:
            self.r = r
        else:
            self.r = (
                np.arange(1, self.avg_img.shape[0] // 2)
                * self.radius
                / (self.avg_img.shape[0] // 2)
            )  # unbinned radius variable
        # if self.r not equal to profile_splits.shape[2], print warning
        if self.Cr_profile_splits is not None:
            if len(self.r) != self.Cr_avg_profiles[0].shape[0]:
                print("Warning: r and profile_splits are different lengths.")
        # Initialize optional attributes to empty
        self.Cr_covmat_full = []
        self.Cr_cormat_full = []
        self.Cr_errors_full = []
        self.Cr_covmat_binned = []
        self.Cr_cormat_binned = []
        self.Cr_errors_binned = []  # errors on the binned profile
        self.Sr_covmat_full = []
        self.Sr_cormat_full = []
        self.Sr_errors_full = []
        self.Sr_covmat_binned = []
        self.Sr_cormat_binned = []
        self.Sr_errors_binned = []  # errors on the binned profile

    def set_split_wgts(self, additional_weights=None):
        # optionally replace split_wgts
        # if additional_weights is None, weights depend only on number of peaks in each split
        if additional_weights is None:
            additional_weights = np.ones(self.Nsamples)
        self.split_wgts = (
            self.Npks_splits / np.average(self.Npks_splits) * additional_weights
        )

    def set_average_profiles(
        self,
    ):  # Option to call this by hand, to reset the profile, if the weights have changed
        self.Cr_avg_profiles = []
        self.Sr_avg_profiles = []
        for m, Crprofsplits in enumerate(self.Cr_profile_splits):
            self.Cr_avg_profiles.append(
                np.average(Crprofsplits, axis=0, weights=self.split_wgts)
            )
        for m, Srprofsplits in enumerate(self.Sr_profile_splits):
            self.Sr_avg_profiles.append(
                np.average(Srprofsplits, axis=0, weights=self.split_wgts)
            )

    def set_avg_profiles_binned(self, binsize):
        # a list of the average binned profiles for each multipole moment m. Length m_max, each element shape (n_bins,)
        self.Cr_avg_profiles_binned = []
        for m, avgprof in enumerate(self.Cr_avg_profiles):
            binned_prof, binned_r = ss.bin_profile(
                self.r, avgprof, self.radius, binsize
            )
            self.Cr_avg_profiles_binned.append(np.asarray(binned_prof))
        self.Sr_avg_profiles_binned = []
        for m, avgprof in enumerate(self.Sr_avg_profiles):
            binned_prof, binned_r = ss.bin_profile(
                self.r, avgprof, self.radius, binsize
            )
            self.Sr_avg_profiles_binned.append(np.asarray(binned_prof))
        self.r_binned = np.asarray(
            binned_r
        )  # set binned r as whatever the last binned r was. These should all be the same.

    def set_custom_bin_m_avg(self, m, custom_bins):
        # rebin the mth multipole moment of the profiles
        custom_bins = np.asarray(custom_bins)
        custom_Cr_profile_m = [
            np.average(self.Cr_avg_profiles[m][custom_bins[i] : custom_bins[i + 1]])
            for i, bin in enumerate(custom_bins[:-1])
        ]
        self.Cr_avg_profiles_binned[m] = np.asarray(custom_Cr_profile_m)
        custom_Sr_profile_m = [
            np.average(self.Sr_avg_profiles[m][custom_bins[i] : custom_bins[i + 1]])
            for i, bin in enumerate(custom_bins[:-1])
        ]
        self.Sr_avg_profiles_binned[m] = np.asarray(custom_Sr_profile_m)
        self.r_binned = (
            (custom_bins[:-1] + custom_bins[1:])
            / 2
            * self.radius
            / (self.avg_img.shape[0] // 2)
        )

    def set_profile_splits_binned(self, binsize):  # bin the profile of each split
        if not self.__has_splits__:
            print("No splits to bin.")
            return
        else:
            profile_splits_binned = []
            for m, profsplits in enumerate(self.Cr_profile_splits):
                profile_splits_binned_m = []
                for split in profsplits:
                    binned_prof, binned_r = ss.bin_profile(
                        self.r, split, self.radius, binsize
                    )
                    profile_splits_binned_m.append(np.asarray(binned_prof))
                profile_splits_binned.append(np.asarray(profile_splits_binned_m))
            self.Cr_profile_splits_binned = profile_splits_binned  # list with len(m_max), each element shape (n_splits, n_bins)
            # repeat for Sr
            for m, profsplits in enumerate(self.Sr_profile_splits):
                profile_splits_binned_m = []
                for split in profsplits:
                    binned_prof, binned_r = ss.bin_profile(
                        self.r, split, self.radius, binsize
                    )
                    profile_splits_binned_m.append(np.asarray(binned_prof))
                profile_splits_binned.append(np.asarray(profile_splits_binned_m))
            self.Sr_profile_splits_binned = profile_splits_binned  # list with len(m_max), each element shape (n_splits, n_bins)
            # not making into array because each element may have different shape after reassignment; see set_custom_bin_m

    def set_custom_bin_m_splits(self, m, custom_bins):
        # rebin the mth multipole moment of the profiles
        if not self.__has_splits__:
            print("No splits to bin.")
            return
        else:
            custom_profile_m = [
                np.average(
                    self.Cr_profile_splits[m][:, custom_bins[i] : custom_bins[i + 1]],
                    axis=1,
                )
                for i, bin in enumerate(custom_bins[:-1])
            ]
            self.Cr_profile_splits_binned[m] = np.asarray(custom_profile_m).transpose()
            # repeat for Sr
            custom_profile_m = [
                np.average(
                    self.Sr_profile_splits[m][:, custom_bins[i] : custom_bins[i + 1]],
                    axis=1,
                )
                for i, bin in enumerate(custom_bins[:-1])
            ]
            self.Sr_profile_splits_binned[m] = np.asarray(custom_profile_m).transpose()

    def set_covariance_full(self):
        # set the covariance matrix for the full profile
        if not self.__has_splits__:
            print("No splits to bin.")
            return
        else:
            for m, profsplits in enumerate(self.Cr_profile_splits):
                covmat, cormat = ss.covariances(
                    profsplits, self.split_wgts, self.Nsamples
                )
                self.Cr_covmat_full.append(covmat)
                self.Cr_cormat_full.append(cormat)
                self.Cr_errors_full.append(np.sqrt(np.diag(covmat)))
            # same for Sr
            for m, profsplits in enumerate(self.Sr_profile_splits):
                covmat, cormat = ss.covariances(
                    profsplits, self.split_wgts, self.Nsamples
                )
                self.Sr_covmat_full.append(covmat)
                self.Sr_cormat_full.append(cormat)
                self.Sr_errors_full.append(np.sqrt(np.diag(covmat)))

    def set_covariance_binned(self):
        # set the covariance matrix for the binned profile
        # reset in case already set
        self.Cr_covmat_binned = []
        self.Cr_cormat_binned = []
        self.Cr_errors_binned = []
        self.Sr_covmat_binned = []
        self.Sr_cormat_binned = []
        self.Sr_errors_binned = []
        if not self.__has_splits__:
            print("No splits to bin.")
            return
        else:
            for m, profsplits in enumerate(self.Cr_profile_splits_binned):
                covmat, cormat = ss.covariances(
                    profsplits, self.split_wgts, self.Nsamples
                )
                self.Cr_covmat_binned.append(covmat)
                self.Cr_cormat_binned.append(cormat)
                self.Cr_errors_binned.append(np.sqrt(np.diag(covmat)))
            for m, profsplits in enumerate(self.Sr_profile_splits_binned):
                covmat, cormat = ss.covariances(
                    profsplits, self.split_wgts, self.Nsamples
                )
                self.Sr_covmat_binned.append(covmat)
                self.Sr_cormat_binned.append(cormat)
                self.Sr_errors_binned.append(np.sqrt(np.diag(covmat)))

    def bin_and_get_stats(self, binsize):
        self.set_profile_splits_binned(binsize)
        self.set_avg_profiles_binned(binsize)
        self.set_covariance_full()
        self.set_covariance_binned()


def retrieve_stack_info(
    path,
    format="constant_comoving",
    binsize=2,
    crop_center=2.0,
    r_m0_normalization=None,
    mapchoice='default'
):
    """
    Args:
    path:
    format (str): options are 'constant_comoving' [Mpc], 'constant_physical' [Mpc], 'constant_angular' [deg]
    crop_center (float): the radius (in units matching 'format' choice) within which to crop out the profiles
    """
    import h5py as h5

    print(f"retrieving data from {path}")
    if format == "constant_comoving":
        stype = "cutout_rad_cMpc"
        stacktype = "stack_comov"
    elif format == "constant_physical":
        stype = "cutout_rad_pMpc"
        stacktype = "stack_phys"
    elif format == "constant_angular":
        stype = "cutout_rad_deg"
        stacktype = "stack_deg"
    with h5.File(path, "r") as f:
        for map in f.keys():
            print("Stacks from the following maps available:", map)
        if mapchoice=='default':
            for map in f.keys():
                mapdata = f[map]
                print("Reading stack data from the map:", map)
                break # just take the first map
        
        cutout_rad = f.attrs[stype]
        imgs = []
        wgts = []
        Crprofs = []
        Srprofs = []
        for reg in mapdata.keys():
            thisreg_imgs = []
            thisreg_wgts = []
            for zbin in mapdata[reg].keys():
                if np.any(np.isnan(mapdata[reg][zbin][stacktype][:])):
                    print(
                        "NaN detected in region",
                        reg,
                        "redshift bin",
                        zbin,
                        "; skipping this bin.",
                    )
                else:
                    thisreg_imgs.append(mapdata[reg][zbin][stacktype][:])
                    thisreg_wgts.append(mapdata[reg][zbin].attrs["Nobj"])
                
                    
            thisreg_stack = np.average(
                np.asarray(thisreg_imgs), weights=thisreg_wgts, axis=0
            )
            imgs.append(thisreg_stack)
            wgts.append(mapdata[reg].attrs["Nobj"])
            print("Radial decompose region:", reg)
            r, Cr, Sr = ss.radial_decompose_2D(thisreg_stack, 5, f.attrs[stype])
            Crprofs.append(Cr)
            Srprofs.append(Sr)
    Crprofs = np.array(Crprofs).transpose(1, 0, 2)
    Srprofs = np.array(Srprofs).transpose(1, 0, 2)
    if crop_center is not None:
        idx_crop = np.where(np.abs(r - crop_center) == np.min(np.abs(r - crop_center)))[
            0
        ][0]
        Crprofs = Crprofs[:, :, idx_crop:]
        Srprofs = Srprofs[:, :, idx_crop:]
        r = r[idx_crop:]  # reset r to account for cut
    MyStack = Stack(
        radius=cutout_rad,
        img_splits=np.asarray(imgs),
        Cr_profile_splits=Crprofs,
        Sr_profile_splits=Srprofs,
        Npks_splits=wgts,
        r=r,
    )

    if r_m0_normalization is not None:
        idx_rnorm = np.where(
            np.abs(MyStack.r - r_m0_normalization)
            == np.min(np.abs(MyStack.r - r_m0_normalization))
        )[0][0]
        for r in range(MyStack.Cr_profile_splits.shape[1]):
            avg_Cr_norm = np.average(MyStack.Cr_profile_splits[0, r, :][idx_rnorm:])
            MyStack.Cr_profile_splits[0, r, :] -= avg_Cr_norm

    MyStack.set_average_profiles()
    MyStack.bin_and_get_stats(binsize)  # Mpc
    return MyStack

def plotstack(im_array, radius, vmin=-1e-7, vmax=1e-7, smooth=False, unit='cMpc', label="Compton-$y$", grid=True, title=None, subtract_average=False):
    from scipy import ndimage
    import matplotlib.pyplot as plt
    fig    = plt.figure(figsize=[8,5])
    if smooth:
        toplot = ndimage.gaussian_filter(im_array, sigma=8)
    else:
        toplot = im_array
    if subtract_average:
        toplot = toplot - get_annulus(im_array)
    smoothplot = plt.imshow(toplot, origin='lower', cmap='afmhot', vmin=vmin, vmax=vmax)
    imhalf = im_array.shape[0]//2
    if grid:
        plt.grid()
        
        plt.axvline(imhalf, color='k')
        plt.axhline(imhalf, color='k')

    N = im_array.shape[0]
    locs = np.linspace(0, N - 1, 9)

    
    units_per_pix = radius / imhalf
    labels = ["{:.1f}".format((l - imhalf) * units_per_pix) for l in locs]


    plt.xlabel(f"x [{unit}]")
    plt.ylabel(f"y [{unit}]")
    plt.xticks(locs, labels)
    plt.yticks(locs, labels)
    if title is not None:
        plt.title(title)
    cbar = fig.colorbar(smoothplot)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.set_label(label)
    cbar.update_ticks()
    return toplot


def get_annulus(image):
    center = [int(image.shape[0]/2), int(image.shape[1]/2)]
    # R1 and R2 vary with image size
    # R1 is 1/2 of way from center of image
    R1   = int(center[0]/2.)
    # R2 is 3/4 of way from center of image
    R2   = int(3*center[1]/4.)
    imin = center[0] - R2
    imax = center[0] + R2 + 1
    jmin = center[1] - R2
    jmax = center[1] + R2 + 1
    target = []
    for i in np.arange(imin, imax):
        for j in np.arange(jmin, jmax):
            ij = np.array([i,j])
            dist = np.linalg.norm(ij - np.array(center))
            if dist > R1 and dist <= R2:
                target.append(image[i][j])
    target = np.array(target)
    return np.average(target)