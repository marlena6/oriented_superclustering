from pixell import reproject, enmap, utils, enplot
import numpy as np
from scipy.interpolate import RectBivariateSpline

def stackChunk(Chunk, imap, cutout_size_deg, cutout_resolution_deg, orient):
    # extract sample postage stamp around 0,0. Use this because thumbnail_geometry not working
    thumb_base = reproject.thumbnails(imap, coords=np.deg2rad([0,0]).T, res=cutout_resolution_deg*utils.degree, r=cutout_size_deg*utils.degree, method="spline", order=1)
    thumb_shape, thumb_wcs = thumb_base.shape, thumb_base.wcs
    # get the thumbnails
    # thumb_shape, thumb_wcs = enmap.thumbnail_geometry(r=cutout_size_deg*utils.degree, res=cutout_resolution_deg*utils.degree) # should eventually use either utils or u for units, not both
    resMap = enmap.zeros(thumb_shape, thumb_wcs) # initialize
    # radian positions of each pixel
    ipos = resMap.posmap()
    X = ipos[0]
    Y = ipos[1] #[:, ::-1] # flipping the order for use in scipy later
    XY = np.array([X.flatten(), Y.flatten()])

    # size of canvas in radians (this is for just x direction, but cutout is symmetric)
    size = ipos[0,:,:].max() - ipos[0,:,:].min()

    # size of pixel in radians
    dx = float(size) / (resMap.shape[0]-1)
    dy = float(size) / (resMap.shape[1]-1)

    # centers of the pixels (index+0.5 times size)
    x_bins = dx * (np.arange(resMap.shape[0]+1) - 0.5)
    y_bins = dy * (np.arange(resMap.shape[1]+1) - 0.5)
    x_grid, y_grid = np.meshgrid(x_bins, y_bins, indexing='ij')

    x_grid = x_grid*180./np.pi*60.
    y_grid = y_grid*180./np.pi*60.
    cell_size = x_grid[1, 0]-x_grid[0, 0]
    x_grid += cell_size/2.
    y_grid += cell_size/2.
    x_grid = x_grid[:-1, :-1]
    y_grid = y_grid[:-1, :-1]
    x_grid -= (x_grid.max()-x_grid.min())/2.
    y_grid -= (y_grid.max()-y_grid.min())/2.

    r = np.sqrt(x_grid**2+y_grid**2)
    th = np.arctan2(y_grid, x_grid)
    th[th < 0.] += 2.*np.pi

    r_bins = np.linspace(1., np.floor(x_grid.max()), 11)#6)
    r_binc = (r_bins[:-1] + r_bins[1:]) / (2.0)

    hist_norm, _ = np.histogram(r.flatten(), bins=r_bins)

    count = 0
    want_random = False # TESTING
    if want_random:
        seed_def = 6000 # def
        seed = 3000
        np.random.seed(seed) # randomized
    # get thumbnails for all the objects in the chunk
    ra = Chunk.RA  # in deg
    dec = Chunk.DEC # in deg
    # extract postage stamps around the objects. Need them to be larger than the cutout size (I think?)
    thumbs = reproject.thumbnails(imap, coords=np.deg2rad([dec,ra]).T, res=cutout_resolution_deg*utils.degree, r=cutout_size_deg*utils.degree+ 0.5*utils.degree, method="spline", order=1)
    # radian positions of each pixel
    ipos_lrg = thumbs[0].posmap()
    X_lrg = ipos_lrg[0]
    Y_lrg = ipos_lrg[1][:, ::-1] # flipping the order for use in scipy later
    x_lrg, y_lrg = X_lrg[:, 0], Y_lrg[0, :]
    print("x_lrg shape is", x_lrg.shape, "y_lrg shape is", y_lrg.shape, "thumbnail shape is", thumbs[0].shape)
    for iObj in range(Chunk.nObj):
        if iObj%10000==0:
            print("- analyze object", iObj)
        # if ts.overlapFlag[iObj]: # Re-implement this later
        
        # randomized
        if orient=='random':
            alpha = np.random.rand()*2.*np.pi
            ca = np.cos(alpha)
            sa = np.sin(alpha)
        else:
            ca = np.cos(Chunk.alpha[iObj]) # cos(alpha)
            sa = np.sin(Chunk.alpha[iObj]) # sin(alpha)
            # show the thumbnail
            # enplot.show(enplot.plot(thumbs[iObj], colorbar=True, ticks=20))
        
            fun2D = RectBivariateSpline(x_lrg, y_lrg, thumbs[iObj], kx=1, ky=1)
            # R = np.array([[ca, sa], [-sa, ca]]) # tested that this is the right
            R = np.array([[ca, -sa], [sa, ca]]) # TESTING!!!! I think mirror reflected
            X_rot, Y_rot = np.dot(R, XY)
            stampMap = fun2D(X_rot, Y_rot, grid=False).reshape(resMap.shape)
            del X_rot, Y_rot
        resMap += (stampMap)

        count += 1
    

    # # dispatch each chunk of objects to a different processor
    # with sharedmem.MapReduce(np=ts.nProc) as pool:
    # resMap = np.array(pool.map(stackChunk, list(range(nChunk))))

    # # sum all the chunks
    # resMap = np.sum(resMap, axis=0)
    # # normalize by the proper sum of weights
    # resMap *= norm

    return resMap

class Chunk:
    def __init__(self, nObj, RA, DEC, alpha):
        self.nObj = nObj
        self.RA = RA
        self.DEC = DEC
        self.alpha = alpha
    