import numpy as np
from astropy.io import fits
import sys
from astropy.cosmology import z_at_value
import gc
import healpy as hp
import astropy.units as u

def get_radecz(filepath, return_id=False, return_weight=False):
    if "desi" in filepath: # DESI LRG file
        print("DESI LRG catalog entered.")
        hdu = fits.open(filepath)
        dat = hdu[1].data
        hdu.close()
        ra  = dat['RA']
        dec = dat['DEC']
        z   = dat['Z']
        id  = dat['TARGETID']
        try:
            w   = dat['WEIGHT']
        except:
            w = dat['w']
    elif "flam" in filepath:
        print("Flamingo catalog entered.")
        with np.load(filepath) as data:
            ra = data['ra']
            dec = data['dec']
            z  = data['z']
            id = data['sub_idx']
            w  = np.ones(len(ra))
    else:
        print("unrecognized file format")
    to_return = [ra,dec,z]
    if return_id:
        to_return.append(id)
    if return_weight:
        to_return.append(w)
    return to_return

def delta_g(nside, ra, dec, ra_rand=None, dec_rand=None, catalog_weights=None, randoms_weights=None, alpha=1, smth=0, beam='gaussian'): # smoothing scale in arcsec
    '''
    Get a number density map from a set of coordinates and weights
    Parameters:
    nside: int
        Healpix nside parameter
    theta: float array
        Declination in radians
    phi: float array
        Right ascension in radians
    binmask: float array
        Array of values to mask the map with (must be binary)
    fracmask: float array
        Array of values to correct the map for survey incompleteness
    smth: float
        Smoothing scale in arcminutes
    wgt: float array
        Array of weights for each coordinate
    beam: string
        Beam type for smoothing. Options are 'gaussian' or 'tophat'
    Returns:
    map: float array
        Number density map (smoothed or unsmoothed)
    '''
    import healpy as hp
    threshold=1e-5
    if catalog_weights is None:
       catalog_weights = np.ones_like(ra)
    if randoms_weights is None:
       randoms_weights = np.ones_like(ra_rand)
        
    data_map   = np.zeros((hp.nside2npix(nside)))
    pix = hp.ang2pix(nside, ra, dec, lonlat=True)
    np.add.at(data_map, pix, catalog_weights)
    rand_map = np.zeros((hp.nside2npix(nside)))
    rand_pix = hp.ang2pix(nside, ra_rand, dec_rand, lonlat=True)
    np.add.at(rand_map, rand_pix, randoms_weights)
    print("sum data", np.sum(data_map), "sum rand", np.sum(rand_map))

    hp.mollview(rand_map, max=.005)
    hp.mollview(data_map, max=.005)
    alpha = np.sum(catalog_weights)/np.sum(randoms_weights)
    print("alpha is", alpha)

    delta_map = np.zeros(data_map.size)
    smoothed_data_map = hp.sphtfunc.smoothing(data_map, fwhm = np.deg2rad(smth/60.), pol=False)
    smoothed_rand_map = hp.sphtfunc.smoothing(rand_map, fwhm = np.deg2rad(smth/60.), pol=False)
    smoothed_diff_map = smoothed_data_map - alpha * smoothed_rand_map
    
    threshold= 0.2 * np.mean(smoothed_rand_map)
    print("Min of randoms within threhsold is", np.min(smoothed_rand_map[smoothed_rand_map>threshold]))
    is_observed = abs(smoothed_rand_map) > threshold
    print("Mean smoothed diff", np.mean(smoothed_diff_map[is_observed]), "Mean smoothed rand", np.mean(smoothed_rand_map[is_observed]), "Mean smoothed data", np.mean(smoothed_data_map[is_observed]), "ratio data to rand with alpha", np.mean(smoothed_data_map[is_observed])/(alpha*np.mean(smoothed_rand_map[is_observed])))
    delta_map[is_observed] = smoothed_diff_map[is_observed] / smoothed_rand_map[is_observed]
    # define the mask as wherever the randoms counts are below the threshold
    print("Mean data:", np.mean(data_map[is_observed]))
    print("Mean rand:", np.mean(rand_map[is_observed]))
    print("Mean of delta map is", np.mean(delta_map[is_observed]))
    hp.mollview(delta_map, max=0.5, min=-0.5)
    return delta_map

def DecRatoThetaPhi(dec,ra):
    theta = np.deg2rad(-dec+90.)
    phi = np.deg2rad(ra)
    return theta,phi

def ThetaPhitoRaDec(theta,phi,negative_ras=False):
    ra = np.rad2deg(phi)
    if negative_ras:
        ra[ra>180] = -360 + ra[ra>180] # convert RAs greater than 180 to negative values
    dec = -1*(np.rad2deg(theta)-90.)
    return ra,dec

def dlist(cosmo, minz=None, maxz=None, slice_width=None, offset=0, zlist=None, dlist=None):
    if zlist is None and minz and dlist is None:
        sys.exit("Need to input one of either zbins, minz, dlist.")
    if zlist is not None:
        dlist = []
        for z in zlist:
            # limit my sample to objects which exist in this redshift bin
            dist_slice_min, dist_slice_max = cosmo.comoving_distance(z[0]),cosmo.comoving_distance(z[1])
            dlist.append([int(dist_slice_min.value), int(dist_slice_max.value)])
    elif dlist is not None:
        dlist = np.asarray(dlist)
        zlist = [[z_at_value(cosmo.comoving_distance, d[0]*u.Mpc).value, z_at_value(cosmo.comoving_distance, d[1]*u.Mpc).value] for d in dlist]
        print(zlist)
    else:
        dlist = []
        nbins = int((cosmo.comoving_distance(maxz).value - cosmo.comoving_distance(minz).value) // slice_width)
        print("Number of distance bins: %d" %nbins)
        for i in range(nbins):
            dist_slice_min = cosmo.comoving_distance(minz)+float(offset)*u.Mpc + slice_width*u.megaparsec*i
            dist_slice_max = dist_slice_min + slice_width*u.megaparsec
            dlist.append([int(dist_slice_min.value), int(dist_slice_max.value)])
        zlist = z_at_value(cosmo.comoving_distance, [d[0]*u.Mpc for d in dlist]).value
    return dlist, zlist

def overdensity_to_potential(overdensity_map_alms, nside):
    # ell filter
    ls = np.arange(3*nside)
    ls[0] = 1.
    llplus = -1./(ls*(ls+1))
    llplus[0] = 0.
    # apply
    potential_alms = hp.sphtfunc.almxfl(overdensity_map_alms, llplus)
    potential_map  = hp.sphtfunc.alm2map(potential_alms, nside) # rewrite inmap as potential map
    return potential_alms, potential_map

    
def tidal_field(alms, nside, cotth, return_grads=True):
    npix = hp.nside2npix(nside)
    
    # first derivative (dphi/dtheta and dphi/sin(theta)dphi)
    dphitheta_map, dphisphi_map = hp.alm2map_der1(alms, nside)[-2:]

    # turn into alms again (no direct function maybe because numerical)
    dphitheta = hp.sphtfunc.map2alm(dphitheta_map, pol=False)
    dphisphi = hp.sphtfunc.map2alm(dphisphi_map, pol=False)

    # t11 is derivative with respect to theta twice (dphithetatheta)
    t11, dphisphitheta = hp.alm2map_der1(dphitheta, nside)[-2:]

    # t21 is derivative with respect to phi and then theta
    t21, dphisphisphi = hp.alm2map_der1(dphisphi, nside)[-2:]
    t12 = t21[:]
    
    # t22 is a mixture of stuff
    dphitheta = hp.sphtfunc.alm2map(dphitheta, nside)
    t22 = cotth*dphitheta + dphisphisphi
    if not return_grads:
        del dphisphitheta, dphisphisphi, dphitheta, dphisphi, cotth
    else: del dphisphitheta, dphisphisphi, cotth
    gc.collect()
    
    # to be used as a sanity check
    delta = t22 + t11

    # creating the tidal tensor
    tidal = np.zeros((npix, 2, 2))
    tidal[:, 0, 0] = t11
    tidal[:, 0, 1] = t12
    tidal[:, 1, 0] = t21
    tidal[:, 1, 1] = t22
    del t11, t12, t21, t22
    gc.collect()
    if return_grads:
        return tidal, dphitheta_map, dphisphi_map
    else:
        return tidal

def measure_orientation(ra, dec, overdensity_map, cotth, e_min=None, e_max=None, nu_min=None, mode='density', return_xy_pol=True):
    # standard check: ensure zero mean
    assert np.abs(np.mean(overdensity_map)) < 1e-3, "The input map does not have zero mean."
    
    nside = hp.get_nside(overdensity_map)
    alms  = hp.sphtfunc.map2alm(overdensity_map, pol=False)
    
    if mode=='potential': # convert the density to a potential field
        alms, inmap = overdensity_to_potential(alms, nside)
    else:
        inmap = overdensity_map
    if return_xy_pol:
        tidal, dphitheta, dphisphi = tidal_field(alms, nside, cotth)
    else:
        tidal = tidal_field(alms, nside, cotth) # tidal is shape pix, 2, 2
    ### selections ###
    # find pixel indices for each object
    
    pix = hp.ang2pix(nside, ra, dec, lonlat=True)
    tidal_obj = tidal[pix,:,:] # nobj, 2, 2
    evals, evecs = np.linalg.eig(tidal_obj) # nobj, 2 (evals), nobj, 2, 2 (evecs)
    evals *= -1 # reverse sign so that peaks are positive. Note: different than Boryana's implementation
    i_sort = np.argsort(evals, axis=1) # same shape as evals
    print(evals[i_sort].shape)
    if e_min is not None or e_max is not None:
        # compute ellipticity for the objects: lambda1-lambda2/2(lambda1+lambda2)
        eigs_larger = np.take_along_axis(evals,i_sort, axis=1)[:,1]
        eigs_smaller = np.take_along_axis(evals,i_sort, axis=1)[:,0]
        e = (eigs_larger-eigs_smaller)/(2*(eigs_larger+eigs_smaller)+5e-8)
        if e_min is not None:
            ecut_min = e > e_min
        else:
            ecut_min = np.ones(len(e), dtype=bool)
        if e_max is not None:
            ecut_max = e < e_max
        else:
            ecut_max = np.ones(len(e), dtype=bool)
    if e_min is None and e_max is None:
        ecut_min = np.ones(len(evals), dtype=bool)
        ecut_max = np.ones(len(evals), dtype=bool)
    if nu_min is not None:
        # compute nu: delta/sigma
        sigma = np.std(inmap)
        print("Computed rms of the field: {:.4f}".format(sigma))
        nu_obj = inmap[pix]/sigma
        nucut_min = nu_obj > nu_min
    else:
        nucut_min = np.ones(len(evals), dtype=bool)
    # combine the boolean cuts
    final_cut = ecut_min & ecut_max & nucut_min
    
    ### selections ###
    
    ### orientation ###

    evals_sorted = np.zeros((np.sum(final_cut), 2))
    evecs_sorted = np.zeros((np.sum(final_cut), 2, 2))
    evals_sel = evals[final_cut]
    evecs_sel = evecs[final_cut]
    i_sort_sel = i_sort[final_cut]
    for i in range(evals_sel.shape[0]):
        evecs_sorted[i, :, 0] = evecs_sel[i, :, i_sort_sel[i, 0]]
        evecs_sorted[i, :, 1] = evecs_sel[i, :, i_sort_sel[i, 1]]
        evals_sorted[i, 0] = evals_sel[i, i_sort_sel[i, 0]]
        evals_sorted[i, 1] = evals_sel[i, i_sort_sel[i, 1]]
    del i_sort, i_sort_sel
    gc.collect()
    del evals, evecs, evals_sel, evecs_sel
    gc.collect()
    evals, evecs = evals_sorted, evecs_sorted
    e_th = np.array([1., 0.]) # A e_theta unit vector
    ca = np.zeros(evals.shape[0])
    sa = np.zeros(evals.shape[0])
    alpha = np.zeros(evals.shape[0])
    if return_xy_pol:
        x_pol = np.ones(evals.shape[0])
        y_pol = np.ones(evals.shape[0])
        
    for i in range(evals.shape[0]):
        print(evals[i])
        e2 = evecs[i, :, 1] # B smallest eigenvector
        # assert np.isclose(np.linalg.norm(e2), 1.)
        ca[i] = np.dot(e_th, e2)
        sa[i] = np.cross(e_th, e2) # applying ca, -sa, sa, ca to A gives 1 in the dot product with B for any A, B
        alpha[i] = np.arctan2(sa[i], ca[i])
        if return_xy_pol:
            # measure the gradient along the e2 direction and make sure it's positive (i.e. e2 points "uphill")
            grad_e2 = e2[0]*dphitheta[pix[final_cut][i]] + e2[1]*dphisphi[pix[final_cut][i]]
            if grad_e2<0:
                x_pol[i] = -1
            e1 = evecs[i, :, 1]
            grad_e1 = e1[0]*dphitheta[pix[final_cut][i]] + e1[1]*dphisphi[pix[final_cut][i]]
            if grad_e1<0:
                y_pol[i] = -1
    
    if return_xy_pol:
        return alpha, x_pol, y_pol, ca, sa, final_cut
    else:
        return alpha, ca, sa, final_cut