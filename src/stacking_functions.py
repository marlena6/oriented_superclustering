from pixell import reproject, enmap, utils, enplot, curvedsky as cs
import numpy as np
from scipy.interpolate import RectBivariateSpline
import sys
import matplotlib.pyplot as plt
import os
import healpy as hp
import copy

class Chunk:
    def __init__(self, RA, DEC, alpha=None, x_asym=None, y_asym=None, vR=None, velocity_stack=False):
        if len(RA) != len(DEC):
            sys.exit("RA and Dec must have the same length.")
        self.nObj = len(RA)
        self.RA = RA
        self.DEC = DEC
        self.alpha = alpha
        self.x_asym = x_asym
        self.y_asym = y_asym
        self.vR = vR # B.H.
        self.velocity_stack = True if velocity_stack else False

class StackGeometry:
    def __init__(self, cutout_rad_deg, cutout_resolution_deg):
        # extract sample postage stamp around 0,0.
        thumb_shape, thumb_wcs  = enmap.thumbnail_geometry(r=cutout_rad_deg*utils.degree, res=cutout_resolution_deg*utils.degree) # this gives non-square shape
        
        # thumb = reproject.thumbnails(
        #     imap=imap,   # dummy, just for geometry
        #     coords=np.deg2rad([[0, 0]]),
        #     res=cutout_resolution_deg * utils.degree,
        #     r=cutout_rad_deg * utils.degree,
        #     method="spline",
        #     order=1,
        # )

        self.shape = thumb_shape
        self.wcs = thumb_wcs
        print("shape is", self.shape)
        tmp = enmap.zeros(self.shape, self.wcs)
        ipos = tmp.posmap()
        X, Y = ipos

        XY = np.array([X.flatten(), Y.flatten()])  
        self.XY = XY
        # size of canvas in radians (this is for just x direction, but cutout is symmetric)
        self.size = ipos[0, :, :].max() - ipos[0, :, :].min()
        self.cutout_rad_deg = cutout_rad_deg
        self.cutout_resolution_deg = cutout_resolution_deg

def extractThumbnails(
    iChunk,
    geom,
    imap,
    orient='original'
):
    """Extract thumbnails for a chunk of objects from a catalog."""
    ra = iChunk.RA  # in deg
    dec = iChunk.DEC  # in deg
    if orient == "original":
        # get exact size cutouts
        thumbs = reproject.thumbnails(
        imap,
        coords=np.deg2rad([dec, ra]).T,
        res=geom.cutout_resolution_deg * utils.degree,
        r=geom.cutout_rad_deg * utils.degree,
        method="spline",
        order=1,
    )
    else:
         # get slightly larger cutouts to avoid edge effects when rotating
        thumbs = reproject.thumbnails(
            imap,
            coords=np.deg2rad([dec, ra]).T,
            res=geom.cutout_resolution_deg * utils.degree,
            r=geom.cutout_rad_deg * utils.degree + 0.2 * utils.degree,
            method="spline",
            order=1,
        )
    # # check if any thumbnails contain NaNs or are all 0
    # add an assert statement here later
    # for it, thumb in enumerate(thumbs):
    #     if np.any(np.isnan(np.mean(thumb, (-2, -1)))):
    #         print(f"Thumbnail {it} at RA, dec = {iChunk.RA[it]}, {iChunk.DEC[it]} contains NaNs.")
    #         plt.imshow(thumb)
    #         plt.savefig(f"testing/thumb{it}.png")
    #     elif np.all(thumb == 0):
    #         print(f"Thumbnail {it} at RA, dec = {iChunk.RA[it]}, {iChunk.DEC[it]} is all zeros.")
    #         plt.imshow(thumb)
    #         plt.savefig(f"testing/thumb{it}_zeros.png")
    return thumbs
    
def stackChunk(
    iChunk,
    geom,
    imap,
    orient,
    rescale_1=None,
    rescale_2=None,
    angledef="CofDec",
    thumbnails=None,
    imask=None
):
    """Stack a chunk of objects from a catalog onto an image, with various orientation options.

    Args:
        iChunk (Chunk): Chunk class instance containing RA, DEC, alpha, x_asym, y_asym
        imap (pixell.enmap): Input map to stack onto
        cutout_rad_deg (float): Radius of cutout in degrees
        cutout_resolution_deg (float): Resolution of cutout in degrees
        orient (str): One of 'original', 'random', 'sym', 'asym_x', 'asym_y', or 'asym_xy'
        rescale_1 (float, optional): Scale factor for first rescaled output.
        rescale_2 (float, optional): Scale factor for second rescaled output.
        angledef (str, optional): Which direction & axis the orientation is defined with respect to.
            Defaults to 'CCofRA' (counter clockwise of RA). Other option is 'CofDec' (clockwise of Dec).

    Returns:
        pixell.enmap or tuple: Stacked image, optionally with rescaled versions.
    """
    if not isinstance(iChunk, Chunk):
        sys.exit("iChunk must be an instance of the Chunk class.")
    
    # add a warning about arguments
    if orient not in ["original", "random", "sym", "asym_x", "asym_y", "asym_xy"]:
        sys.exit(
            "orient must be one of 'original', 'random', 'sym', 'asym_x', 'asym_y', or 'asym_xy'."
        )
    if rescale_2 is not None and rescale_1 is None:
        sys.exit("If rescale_2 is specified, rescale_1 must also be specified.")
    
    
    thumb_shape, thumb_wcs = geom.shape, geom.wcs
    # get the thumbnails
    resMap = enmap.zeros(thumb_shape, thumb_wcs)  # initialize
    
    # radian positions of each pixel
    XY = geom.XY
    size = geom.size
    
    # size of pixel in radians
    dx = float(size) / (resMap.shape[0] - 1)
    dy = float(size) / (resMap.shape[1] - 1)

    # centers of the pixels (index+0.5 times size)
    x_bins = dx * (np.arange(resMap.shape[0] + 1) - 0.5)
    y_bins = dy * (np.arange(resMap.shape[1] + 1) - 0.5)
    x_grid, y_grid = np.meshgrid(x_bins, y_bins, indexing="ij")

    x_grid = x_grid * 180.0 / np.pi * 60.0
    y_grid = y_grid * 180.0 / np.pi * 60.0
    cell_size = x_grid[1, 0] - x_grid[0, 0]
    x_grid += cell_size / 2.0
    y_grid += cell_size / 2.0
    x_grid = x_grid[:-1, :-1]
    y_grid = y_grid[:-1, :-1]
    x_grid -= (x_grid.max() - x_grid.min()) / 2.0
    y_grid -= (y_grid.max() - y_grid.min()) / 2.0

    th = np.arctan2(y_grid, x_grid)
    th[th < 0.0] += 2.0 * np.pi

    want_random = False  # TESTING
    if want_random:
        seed = 3000
        np.random.seed(seed)  # randomized
    # get thumbnails for all the objects in the chunk
    ra = iChunk.RA  # in deg
    dec = iChunk.DEC  # in deg
    # extract postage stamps around the objects. Need them to be larger than the cutout size (I think?)
    # print("before all thumbs")
    
    if orient == "original":
        if thumbnails is not None:
            thumbs = thumbnails
        else:
            # get exact size cutouts
            thumbs = reproject.thumbnails(
            imap,
            coords=np.deg2rad([dec, ra]).T,
            res=geom.cutout_resolution_deg * utils.degree,
            r=geom.cutout_rad_deg * utils.degree,
            method="spline",
            order=1,
        )
    else:
        if thumbnails is not None:
            thumbs = thumbnails
        else:
            # get slightly larger cutouts to avoid edge effects when rotating
            thumbs = reproject.thumbnails(
                imap,
                coords=np.deg2rad([dec, ra]).T,
                res=geom.cutout_resolution_deg * utils.degree,
                r=geom.cutout_rad_deg * utils.degree + 0.2 * utils.degree,
                method="spline",
                order=1,
            ) 
    
    # print("after all thumbs")
    # print(f"found {thumbs.shape[0]} thumbnails out of {iChunk.nObj} objects")
    # for it, thumb in enumerate(thumbs):
    #     if np.any(np.isnan(np.mean(thumbs[it],(-2,-1))[...,None,None])):
    #         print(f"Thumbnail {it} at RA, dec = {iChunk.RA[it]}, {iChunk.DEC[it]} contains NaNs.")
    #         plt.imshow(thumb)
    #         plt.savefig(f"testing/thumb{it}.png")
            
    # sys.exit("stop here")
        # show the thumbnail
            # print(f"Object {iObj} is at RA: {iChunk.RA[iObj]}, Dec: {iChunk.DEC[iObj]}, alpha: {iChunk.alpha[iObj]}")
            # plt.imshow(thumbs[iObj])
            # # enplot.show(enplot.plot(thumbs[iObj], colorbar=True, ticks=20))
            # plt.savefig(f"testing/thumb{iObj}.png")
            # plt.clf()
        
    # radian positions of each pixel
    ipos_lrg = thumbs[0].posmap()
    X_lrg = ipos_lrg[0]
    Y_lrg = ipos_lrg[1][:, ::-1]  # flipping the order for use in scipy later
    x_lrg, y_lrg = X_lrg[:, 0], Y_lrg[0, :]

    for iObj in range(iChunk.nObj):
        # if ts.overlapFlag[iObj]: # Re-implement this later

        
        if orient == "original":
            resMap += thumbs[iObj]
        else:
            if orient == "random":  # random orientation angles
                alpha = np.random.rand() * 2.0 * np.pi
                ca = np.cos(alpha)
                sa = np.sin(alpha)
            elif orient in ["sym", "asym_x", "asym_y", "asym_xy"]:
                ca = np.cos(iChunk.alpha[iObj])
                sa = np.sin(iChunk.alpha[iObj])

            
            if angledef == "CCofRA":
                R = np.array([[ca, sa], [-sa, ca]]) # Boryana's version (orientations defined wrt RA axis, I think)
            elif angledef == "CofDec":
                R = np.array(
                    [[ca, -sa], [sa, ca]]
                )  # COOP version (orientations defined wrt Dec axis, but clockwise, I think)
            else:
                sys.exit("angledef must be one of 'CCofRA' or 'CofDec'.")
            stamp = thumbs[iObj]

            # B.H.
            if iChunk.velocity_stack:
                stamp *= iChunk.vR[iObj]
            
            fun2D = RectBivariateSpline(x_lrg, y_lrg, stamp, kx=1, ky=1)
            X_rot, Y_rot = np.dot(R, XY)
            stampMap = fun2D(X_rot, Y_rot, grid=False).reshape(resMap.shape)
            # plt.imshow(stampMap)
            # print('x asym', iChunk.x_asym[iObj], 'y asym', iChunk.y_asym[iObj])
            # plt.show()
            # if iObj > 10:
            #     break
            if orient in ["asym_x", "asym_xy"]:
                if iChunk.x_asym[iObj] == 1:
                    stampMap = np.fliplr(stampMap)

            if orient in ["asym_y", "asym_xy"]:
                if iChunk.y_asym[iObj] == 1:
                    stampMap = np.flipud(stampMap)
            # plt.imshow(stampMap)
            # plt.show()
            del X_rot, Y_rot
            resMap += stampMap

    resMap = resMap / iChunk.nObj
    nreturn = 1
    # rescale if desired
    if rescale_1 is not None:
        if rescale_1 != 1:
            resMap1 = rescale_img(resMap, resMap.shape[0], rescale_1)
        else:
            resMap1 = resMap
        nreturn += 1
    if rescale_2 is not None:
        if rescale_2 != 1:
            resMap2 = rescale_img(resMap, resMap.shape[0], rescale_2)
        else:
            resMap2 = resMap
        nreturn += 1
    if nreturn == 1:
        return resMap
    elif nreturn == 2:
        return (resMap, resMap1)
    elif nreturn == 3:
        return (resMap, resMap1, resMap2)

    # # dispatch each chunk of objects to a different processor
    # with sharedmem.MapReduce(np=ts.nProc) as pool:
    # resMap = np.array(pool.map(stackChunk, list(range(nChunk))))

    # # sum all the chunks
    # resMap = np.sum(resMap, axis=0)
    # # normalize by the proper sum of weights
    # resMap *= norm


def rescale_img(img, base_sidelen, ratio_to_base):
    """Crop an input image given a ratio, and rescale the result to match a base image
    Args:
    img (np.ndarray): 2D array of input image data
    base_sidelen (int): Side length of base image
    ratio_to_base (float): Ratio of
    """
    from PIL import Image

    # for images centered on further distances, the same radius in degrees
    # corresponds to a larger transverse physical radius. Trim these images
    # to the same radius in Mpc as the first, then resize the array.
    assert img.shape[0] == img.shape[1], "Image must be square"
    img_sidelen = img.shape[0]
    resized_sidelen = ratio_to_base * img_sidelen
    PilImg = Image.fromarray(img)
    adjust = ((img_sidelen - resized_sidelen) / 2.0).value
    CroppedImg = PilImg.crop(
        (adjust, adjust, img_sidelen - adjust, img_sidelen - adjust)
    )
    pil_img_rs = CroppedImg.resize((base_sidelen, base_sidelen))
    resized = np.array(pil_img_rs)
    return resized


def rescale_prof(prof_arr, r, base_r):
    """Crop an input image given a ratio, and rescale the result to match a base image
    Args:
    img (np.ndarray): 2D array of input image data
    base_sidelen (int): Side length of base image
    ratio_to_base (float): Ratio of
    """
    from scipy.interpolate import interp1d

    # for profiles centered on further distances, the same radius in degrees
    # corresponds to a larger transverse physical radius. Trim these profiles
    # to the same radius in desired units as the first, then resize the array.
    prof_func = interp1d(r, prof_arr, axis=1)
    resized = prof_func(base_r)
    return resized

def readmap(inmap_dict, kappa_filter_path="/global/cfs/projectdirs/act/www/dr6_lensing_v1/maps/baseline/kappa_filter_act_dr6_lensing_v1_baseline.txt",
            base_geometry_path = "/global/cfs/projectdirs/act/www/dr6_nilc/ymaps_20230220/ilc_actplanck_ymap.fits"):
    """
    Checks to see if the map is an enamp, and converts from Healpix to Enmap otherwise.
    Args:
        inmap_dict (dict): map dictionary with path to the map file (Healpix or Enmap format), type
    Returns:
        outmap_dict (dict): map dictionary with path to the Enmap file, type
        """
    mappath = inmap_dict["path"]
    outmap_dict = copy.deepcopy(inmap_dict)
    if not os.path.exists(mappath):
        raise ValueError(f"Map path {mappath} does not exist.")
    
    # figure out the format of the map: enmap, healpix, or alms
    if mappath.endswith("fits"):
        try:
            shape, wcs = enmap.read_map_geometry(mappath) # quick geometry check which will fail if not an enmap
            print("Map is already an enmap.")
            outmap_dict["path"] = mappath
        except Exception as e:
            print("Map is not an enmap. Attempting to convert from healpix.")
            enmap_path = mappath.replace(".fits", "_enmap.fits")
            # Check if another map already exists with _enmap in the same directory
            if os.path.exists(enmap_path):
                print("Enmap version of map already exists at {enmap_path}. Using that.")
                outmap_dict["path"] = enmap_path
                if inmap_dict["type"] == 'kappa' and "filter" not in mappath:
                    print("Filtering and reprojecting to enmap. This may take a while...")
                    enmap_path = save_filtered_map(mappath, kappa_filter_path)
            else: # make an enmap
                shape, wcs = enmap.read_map_geometry(base_geometry_path) # use the geometry of the ACT ymap
                hpmap = hp.read_map(mappath)
                if inmap_dict["type"] == "kappa" and "filter" not in mappath:
                    print("Filtering and reprojecting to enmap. This may take a while...")
                    enmap_path = save_filtered_map(mappath, kappa_filter_path, shape=shape, wcs=wcs)
                else:
                    print("Reprojecting healpix map to enmap. This may take a while...")
                    enmap_rp = reproject.healpix2map(hpmap, shape=shape, wcs=wcs, method='spline')
                    enmap.write_map(enmap_path, enmap_rp)
                    print(f"Saved enmap version to {enmap_path}.")
            outmap_dict["path"] = enmap_path
    elif mappath.endswith(".txt"):
        print("Path to map is not a map yet. Assuming it is alms, and attempting to convert alms to map. Will apply ACT kappa almxfl filter.")
        alms = enmap.read_alm(mappath)
        alms = np.nan_to_num(alms, nan=0.0) # turn nans to zeros, they cause troubles
        # use the geometry of the ACT ymap
        shape,wcs = enmap.read_map_geometry(base_geometry_path)
        filter = np.loadtxt(kappa_filter_path)
        filtered_alms = hp.almxfl(alms, filter[:,1])
        outmap = cs.alm2map(filtered_alms, enmap.empty(shape, wcs))
        enmap_path = mappath.replace(".fits", "_filtered_enmap.fits")
        enmap.write_map(enmap_path, outmap)
        print("Kappa map written at", enmap_path)
        # no nans
        assert(not np.any(np.isnan(outmap))), "Error: NaNs found in the kappa map"
        outmap_dict["path"] = enmap_path
    return outmap_dict

def save_filtered_map(map_path, filter_path, shape=None, wcs=None):
    """ Filters a healpix or enmap map with a function and saves the result as an enmap.
    Args:
        map_path (str): Path to the input healpix or enmap map.
        filter_path (str): Path to the filter function (text file with two columns: l and filter(l)).
        shape (tuple): Shape of the output enmap.
        wcs (WCS): WCS of the output enmap.
    """
    filter = np.loadtxt(filter_path)
    if 'enmap' in map_path:
        alms = enmap.map2alm(enmap.read_map(map_path))
        outpath = map_path.replace(".fits", "_filtered.fits")
        shape,wcs = enmap.read_map_geometry(map_path) # replace with the geometry of the input enmap
    else:
        alms = hp.map2alm(hp.read_map(map_path))
        outpath = map_path.replace(".fits", "_enmap_filtered.fits")
    
    filter_kappa = hp.almxfl(alms, filter[:,1])
    outmap = cs.alm2map(filter_kappa, enmap.empty(shape, wcs))
    enmap.write_map(outpath, outmap)
    print(f"Saved filtered kappa map to {outpath}.")
    return outpath