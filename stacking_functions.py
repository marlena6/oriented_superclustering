from pixell import reproject, enmap
import numpy as np
from scipy.interpolate import RectBivariateSpline

def stackChunk(Chunk, imap, cutout_size_deg, cutout_resolution, orient):

    # get the thumbnails
    thumb_shape, thumb_wcs = enmap.thumbnail_geometry(r=cutout_size_deg.value*utils.degree, res=cutout_resolution) # should eventually use either utils or u for units, not both
    resMap = enmap.zeros(thumb_shape, thumb_wcs) # initialize

    # radian positions of each pixel
    ipos = resMap.posmap()
    X = ipos[0]
    Y = ipos[1]
    x, y = X[:, 0], Y[0, :]
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
    for iObj in Chunk.nObj:
        if iObj%10000==0:
            print("- analyze object", iObj)
        if ts.overlapFlag[iObj]:
            # Object coordinates
            ra = Chunk.RA[iObj]  # in deg
            dec = Chunk.DEC[iObj] # in deg
            # extract postage stamp around it. Need it to be larger than the cutout size (I think?) ##MARTINE
            opos, stampMap, stampMask, stampHit = ts.extractStamp(ra, dec, test=False)
            
            # randomized
            if orient=='random':
                alpha = np.random.rand()*2.*np.pi
                ca = np.cos(alpha)
                sa = np.sin(alpha)
            else:
                ca = np.cos(Chunk.alpha[iObj]) # cos(alpha)
                sa = np.sin(Chunk.alpha[iObj]) # sin(alpha)
                fun2D = RectBivariateSpline(x, y, stampMap, kx=1, ky=1)
                R = np.array([[ca, sa], [-sa, ca]]) # tested that this is the right
                #R = np.array([[ca, -sa], [sa, ca]]) # TESTING!!!! I think mirror reflected
                X_rot, Y_rot = np.dot(R, XY)
                stampMap = fun2D(X_rot, Y_rot, grid=False).reshape(resMap.shape)
            del X_rot, Y_rot
        resMap += (stampMap * weightsLong[iObj])

        count += 1
    

    # dispatch each chunk of objects to a different processor
    with sharedmem.MapReduce(np=ts.nProc) as pool:
    resMap = np.array(pool.map(stackChunk, list(range(nChunk))))

    # sum all the chunks
    resMap = np.sum(resMap, axis=0)
    # normalize by the proper sum of weights
    resMap *= norm

    return resMap
