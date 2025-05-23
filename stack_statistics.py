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

    for i in range(1, n):
        r = float(i)
        for j in range(nsteps):
            theta = dtheta * j
            rx = r * np.cos(theta)
            ry = r * np.sin(theta)
            ix = int(min(np.floor(rx), n - 1))
            iy = int(min(np.floor(ry), n - 1))
            rx = rx - ix
            ry = ry - iy
            ix = ix + n
            iy = iy + n
            fv = (1 - rx) * (f[iy, ix] * (1 - ry) + f[iy + 1, ix] * ry) + rx * (
                f[iy, ix + 1] * (1 - ry) + f[iy + 1, ix + 1] * ry
            )
            Cr[0, i - 1] += fv
            for m in range(1, mmax):
                Cr[m, i - 1] = Cr[m, i - 1] + fv * np.cos(m * theta)
                Sr[m, i - 1] = Sr[m, i - 1] + fv * np.sin(m * theta)

    Cr[0, 0 : n - 1] = Cr[0, 0 : n - 1] / nsteps
    Cr[1:mmax, 0 : n - 1] = Cr[1:mmax, 0 : n - 1] * (2.0 / nsteps)
    Sr[1:mmax, 0 : n - 1] = Sr[1:mmax, 0 : n - 1] * (2.0 / nsteps)
    return (r_coords, Cr, Sr)


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
