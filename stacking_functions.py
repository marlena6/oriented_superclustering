from pixell import reproject, enmap, utils
import numpy as np
from scipy.interpolate import RectBivariateSpline
import sys


def stackChunk(
    Chunk,
    imap,
    cutout_rad_deg,
    cutout_resolution_deg,
    orient,
    rescale_1=None,
    rescale_2=None,
):
    # extract sample postage stamp around 0,0.
    # thumb_shape, thumb_wcs  = enmap.thumbnail_geometry(r=cutout_rad_deg*utils.degree, res=cutout_resolution_deg*utils.degree) # this gives non-square shape
    thumb_base = reproject.thumbnails(
        imap,
        coords=np.deg2rad([0, 0]).T,
        res=cutout_resolution_deg * utils.degree,
        r=cutout_rad_deg * utils.degree,
        method="spline",
        order=1,
    )
    thumb_shape, thumb_wcs = thumb_base.shape, thumb_base.wcs
    # get the thumbnails
    resMap = enmap.zeros(thumb_shape, thumb_wcs)  # initialize
    # radian positions of each pixel
    ipos = resMap.posmap()
    X = ipos[0]
    Y = ipos[1]  # [:, ::-1] # flipping the order for use in scipy later
    XY = np.array([X.flatten(), Y.flatten()])

    # size of canvas in radians (this is for just x direction, but cutout is symmetric)
    size = ipos[0, :, :].max() - ipos[0, :, :].min()

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

    count = 0
    want_random = False  # TESTING
    if want_random:
        seed = 3000
        np.random.seed(seed)  # randomized
    # get thumbnails for all the objects in the chunk
    ra = Chunk.RA  # in deg
    dec = Chunk.DEC  # in deg
    # extract postage stamps around the objects. Need them to be larger than the cutout size (I think?)
    thumbs = reproject.thumbnails(
        imap,
        coords=np.deg2rad([dec, ra]).T,
        res=cutout_resolution_deg * utils.degree,
        r=cutout_rad_deg * utils.degree + 0.5 * utils.degree,
        method="spline",
        order=1,
    )  # ('tan')
    # radian positions of each pixel
    ipos_lrg = thumbs[0].posmap()
    X_lrg = ipos_lrg[0]
    Y_lrg = ipos_lrg[1][:, ::-1]  # flipping the order for use in scipy later
    x_lrg, y_lrg = X_lrg[:, 0], Y_lrg[0, :]
    
    for iObj in range(Chunk.nObj):
        # if iObj % 1000 == 0:
        #     print("- analyze object", iObj)
        # if ts.overlapFlag[iObj]: # Re-implement this later

        # need to make sure it works for 'original' orientation too
        if orient != "original":
            # randomized
            if orient == "random":
                alpha = np.random.rand() * 2.0 * np.pi
                ca = np.cos(alpha)
                sa = np.sin(alpha)
            else:
                ca = np.cos(Chunk.alpha[iObj])  # cos(alpha)
                sa = np.sin(Chunk.alpha[iObj])  # sin(alpha)
            # show the thumbnail
            # enplot.show(enplot.plot(thumbs[iObj], colorbar=True, ticks=20))

            fun2D = RectBivariateSpline(x_lrg, y_lrg, thumbs[iObj], kx=1, ky=1)
            # R = np.array([[ca, sa], [-sa, ca]]) # Boryana's version (orientations defined wrt RA axis?)
            R = np.array([[ca, -sa], [sa, ca]])  # COOP version (orientations defined wrt Dec axis?)

            X_rot, Y_rot = np.dot(R, XY)
            stampMap = fun2D(X_rot, Y_rot, grid=False).reshape(resMap.shape)
            if (orient == "asym_x") or (orient == "asym_xy"):
                # add asymmetry
                if Chunk.x_asym[iObj] == 1:
                    stampMap = np.fliplr(stampMap)
            if (orient == "asym_y") or (orient == "asym_xy"):
                if Chunk.y_asym[iObj] == 1:
                    stampMap = np.flipud(stampMap)
            del X_rot, Y_rot
        resMap += stampMap

        count += 1
    resMap = resMap / Chunk.nObj
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


class Chunk:
    def __init__(self, RA, DEC, alpha=None, x_asym=None, y_asym=None):
        if len(RA) != len(DEC):
            sys.exit("RA and Dec must have the same length.")
        self.nObj = len(RA)
        self.RA = RA
        self.DEC = DEC
        self.alpha = alpha
        self.x_asym = x_asym
        self.y_asym = y_asym


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
