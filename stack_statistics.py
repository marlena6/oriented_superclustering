import numpy as np


def bin_profile(r, Cr_m, rad_in_mpc, binsize):  # binsize in Mpc
    # r is in Mpc

    npix = len(Cr_m)
    mpc_per_pix = rad_in_mpc / (npix - 1)
    pix_per_bin = int(np.floor((binsize / mpc_per_pix)))
    binned_prof = []
    binned_r = []
    for i in range(0, npix - pix_per_bin, pix_per_bin):
        binned_prof.append(np.mean(Cr_m[i : i + pix_per_bin]))
        binned_r.append(np.mean(r[i : i + pix_per_bin]))
    # r = [i + 0.5 for i in range(len(binned_prof))]
    # step_in_mpc = rad_in_mpc/len(r)
    return binned_prof, binned_r


def cormat(covmat):
    cormat = np.zeros(covmat.shape)
    for i in range(covmat.shape[0]):
        for j in range(covmat.shape[1]):
            corcoeff = covmat[i, j] / np.sqrt(covmat[i, i] * covmat[j, j])
            cormat[i, j] = corcoeff
    return cormat


def covariances(vec_list, weights, nreg):
    vec_array = np.asarray(vec_list)
    covmat = np.cov(vec_array.T, aweights=weights) / nreg
    correl_mat = cormat(covmat)
    return (covmat, correl_mat)


def radial_decompose_2D(f, mmax, R):
    """
    Args:
    f (np.ndarray): image array
    mmax (int): maximum m for decomposition (maximally 10)
    R (float): half-side-length of image in physical units

    Returns:
    r (np.ndarray): vector of r in units of R
    Cr (np.ndarray): N_m x N_
    """
    n = int(f.shape[0] // 2)
    unit_per_pix = R / (f.shape[0] // 2)
    r_coords = unit_per_pix * np.arange(0, n-1)
    nsteps = n * 20
    dtheta = 2 * np.pi / nsteps
    Cr = np.zeros((mmax, n - 1))
    Sr = np.zeros((mmax, n - 1))
    
    thetas = dtheta * np.arange(nsteps)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    for i in range(1, n):
        
        r = float(i)
        rx = r * cos_t
        ry = r * sin_t
 
        ix = np.minimum(np.floor(rx), n - 1).astype(int)
        iy = np.minimum(np.floor(ry), n - 1).astype(int)
 
        fx = rx - ix
        fy = ry - iy
        
         # Shift indices by +n
        ix += n
        iy += n
        
        # Bilinear interpolation
        f00 = f[iy, ix]
        f01 = f[iy + 1, ix]
        f10 = f[iy, ix + 1]
        f11 = f[iy + 1, ix + 1]
        fv = (1 - fx) * (f00 * (1 - fy) + f01 * fy) + fx * (f10 * (1 - fy) + f11 * fy)

        Cr[0, i - 1] = fv.sum()
        
        # Higher harmonics (vectorized over m)
        m_vals = np.arange(1, mmax)[:, None]  # shape (mmax-1, 1)
        cos_mt = np.cos(m_vals * thetas)
        sin_mt = np.sin(m_vals * thetas)
        
        Cr[1:, i - 1] = (fv * cos_mt).sum(axis=1)
        Sr[1:, i - 1] = (fv * sin_mt).sum(axis=1)
        
    Cr[0, 0 : n - 1] = Cr[0, 0 : n - 1] / nsteps
    Cr[1:mmax, 0 : n - 1] = Cr[1:mmax, 0 : n - 1] * (2.0 / nsteps)
    Sr[1:mmax, 0 : n - 1] = Sr[1:mmax, 0 : n - 1] * (2.0 / nsteps)
    return (r_coords, Cr, Sr)

def CAP_2D(f, R):
    """
    Args:
    f (np.ndarray): image array, one image.
    mmax (int): maximum m for decomposition (maximally 10)
    R (float): half-side-length of image in physical units

    Returns:
    r (np.ndarray): vector of r in units of R
    CAP(r) (np.ndarray): vector of CAP values in units of R
    """
    
    rows, cols = f.shape
    xy_min, xy_max = (-R, R)

    # Generate coordinate values for each axis
    x_coords = np.linspace(xy_min, xy_max, cols)
    y_coords = np.linspace(xy_min, xy_max, rows)

    # pixel_size = (x_coords[1] - x_coords[0])
    # pixArea = (pixel_size)**2

    # Create meshgrid of coordinates
    X, Y = np.meshgrid(x_coords, y_coords)

    # Calculate distances from the center (0,0)
    radial_distances = np.sqrt(X**2 + Y**2)

    r_vals = np.linspace(0, R/np.sqrt(2.0), 20)
    CAP_vals = np.zeros_like(r_vals)
    for i, r in enumerate(r_vals):
        r1 = r * 0.5
    
        inDisk = 1.0 * (radial_distances <= r)
        inRing = 1.0 * (radial_distances > r) * (radial_distances <= r1)
        inRing *= np.sum(inDisk) / np.sum(inRing)
        CAP_vals[i] = float(np.sum((inDisk - inRing) * f))
    return (r_vals, CAP_vals)


def total_multipole_power(
    m_max, img=None, R=None, cos_moments=None, sin_moments=None, r=None
):
    """
    Takes either an image, or pre-computed cosine and sine moments, and returns the total power in each moment
    Args:
    m_max (int): the maximum multipole to compute
    Either:
    img (np.ndarray): the image to decompose
    R (float): the size of the image in physical units

    Or:
    cos_moments (np.ndarray): an N_m x N_r array of the radial integral over cos(m theta),
    where N_m rows correspond to each m and N_r columns correspond to each r bin
    sin_moments (np.ndarray): same as cos, but for the integral of sin(m theta)
    r (np.ndarray): array of floats, the radial coordinate in physical units corresponding to the cos/sin moments

    Returns:
    integrated_power (np.ndarray): a vector length N_m of the integrated power in each moment
    """
    assert (img is not None) or (
        (cos_moments is not None) and (sin_moments is not None)
    ), "Either an image or both the cosine AND sine moments must be passed"
    if cos_moments is None or sin_moments is None:
        assert img is not None, (
            "Both the cosine AND sine moments must be passed if not passing an image"
        )
        r, cos_moments, sin_moments = radial_decompose_2D(img, m_max, R)
    print("cos moments shape", cos_moments.shape)
    m_power_per_r = cos_moments**2 + sin_moments**2
    print("sin moments shape", m_power_per_r.shape)
    integrated_power = np.trapz(m_power_per_r * r, r, axis=1)

    return integrated_power
