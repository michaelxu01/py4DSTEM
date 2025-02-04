# Find the origin of diffraction space

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import leastsq

from ..fit import plane,parabola,bezier_two,fit_2D
from ..utils import get_CoM, add_to_2D_array_from_floats, tqdmnd, get_maxima_2D
from ...io import PointListArray, DataCube
from ..diskdetection.braggvectormap import get_bragg_vector_map
from ..diskdetection.diskdetection import _find_Bragg_disks_single_DP_FK


### Functions for finding the origin


def get_probe_size(DP, thresh_lower=0.01, thresh_upper=0.99, N=100):
    """
    Gets the center and radius of the probe in the diffraction plane.

    The algorithm is as follows:
    First, create a series of N binary masks, by thresholding the diffraction pattern
    DP with a linspace of N thresholds from thresh_lower to thresh_upper, measured
    relative to the maximum intensity in DP.
    Using the area of each binary mask, calculate the radius r of a circular probe.
    Because the central disk is typically very intense relative to the rest of the DP, r
    should change very little over a wide range of intermediate values of the threshold.
    The range in which r is trustworthy is found by taking the derivative of r(thresh)
    and finding identifying where it is small.  The radius is taken to be the mean of
    these r values. Using the threshold corresponding to this r, a mask is created and
    the CoM of the DP times this mask it taken.  This is taken to be the origin x0,y0.

    Args:
        DP (2D array): the diffraction pattern in which to find the central disk.
            A position averaged, or shift-corrected and averaged, DP works best.
        thresh_lower (float, 0 to 1): the lower limit of threshold values
        thresh_upper (float, 0 to 1): the upper limit of threshold values
        N (int): the number of thresholds / masks to use

    Returns:
        (3-tuple): A 3-tuple containing:

            * **r**: *(float)* the central disk radius, in pixels
            * **x0**: *(float)* the x position of the central disk center
            * **y0**: *(float)* the y position of the central disk center
    """
    thresh_vals = np.linspace(thresh_lower, thresh_upper, N)
    r_vals = np.zeros(N)

    # Get r for each mask
    DPmax = np.max(DP)
    for i in range(len(thresh_vals)):
        thresh = thresh_vals[i]
        mask = DP > DPmax * thresh
        r_vals[i] = np.sqrt(np.sum(mask) / np.pi)

    # Get derivative and determine trustworthy r-values
    dr_dtheta = np.gradient(r_vals)
    mask = (dr_dtheta <= 0) * (dr_dtheta >= 2 * np.median(dr_dtheta))
    r = np.mean(r_vals[mask])

    # Get origin
    thresh = np.mean(thresh_vals[mask])
    mask = DP > DPmax * thresh
    x0, y0 = get_CoM(DP * mask)

    return r, x0, y0


def get_origin_single_dp(dp, r, rscale=1.2):
    """
    Find the origin for a single diffraction pattern, assuming (a) there is no beam stop,
    and (b) the center beam contains the highest intensity.

    Args:
        dp (ndarray): the diffraction pattern
        r (number): the approximate disk radius
        rscale (number): factor by which `r` is scaled to generate a mask

    Returns:
        (2-tuple): The origin
    """
    Q_Nx, Q_Ny = dp.shape
    _qx0, _qy0 = np.unravel_index(np.argmax(gaussian_filter(dp, r)), (Q_Nx, Q_Ny))
    qyy, qxx = np.meshgrid(np.arange(Q_Ny), np.arange(Q_Nx))
    mask = np.hypot(qxx - _qx0, qyy - _qy0) < r * rscale
    qx0, qy0 = get_CoM(dp * mask)
    return qx0, qy0


def get_origin(datacube, r=None, rscale=1.2, dp_max=None, mask=None):
    """
    Find the origin for all diffraction patterns in a datacube, assuming (a) there is no
    beam stop, and (b) the center beam contains the highest intensity

    Args:
        datacube (DataCube): the data
        r (number or None): the approximate radius of the center disk. If None (default),
            tries to compute r using the get_probe_size method.  The data used for this
            is controlled by dp_max.
        rscale (number): expand 'r' by this amount to form a mask about the center disk
            when taking its center of mass
        dp_max (ndarray or None): the diffraction pattern or dp-shaped array used to
            compute the center disk radius, if r is left unspecified. Behavior depends
            on type:

                * if ``dp_max==None`` (default), computes and uses the maximal
                  diffraction pattern. Note that for a large datacube, this may be a
                  slow operation.
                * otherwise, this should be a (Q_Nx,Q_Ny) shaped array
        mask (ndarray or None): if not None, should be an (R_Nx,R_Ny) shaped
                    boolean array. Origin is found only where mask==True, and masked
                    arrays are returned for qx0,qy0

    Returns:
        (2-tuple of (R_Nx,R_Ny)-shaped ndarrays): the origin, (x,y) at each scan position
    """
    if r is None:
        if dp_max is None:
            dp_max = np.max(datacube.data, axis=(0, 1))
        else:
            assert dp_max.shape == (datacube.Q_Nx, datacube.Q_Ny)
        r, _, _ = get_probe_size(dp_max)

    qx0 = np.zeros((datacube.R_Nx, datacube.R_Ny))
    qy0 = np.zeros((datacube.R_Nx, datacube.R_Ny))
    qyy, qxx = np.meshgrid(np.arange(datacube.Q_Ny), np.arange(datacube.Q_Nx))

    if mask is None:
        for (rx, ry) in tqdmnd(
            datacube.R_Nx,
            datacube.R_Ny,
            desc="Finding origins",
            unit="DP",
            unit_scale=True,
        ):
            dp = datacube.data[rx, ry, :, :]
            _qx0, _qy0 = np.unravel_index(
                np.argmax(gaussian_filter(dp, r)), (datacube.Q_Nx, datacube.Q_Ny)
            )
            _mask = np.hypot(qxx - _qx0, qyy - _qy0) < r * rscale
            qx0[rx, ry], qy0[rx, ry] = get_CoM(dp * _mask)

    else:
        assert mask.shape == (datacube.R_Nx, datacube.R_Ny)
        assert mask.dtype == bool
        qx0 = np.ma.array(
            data=qx0, mask=np.zeros((datacube.R_Nx, datacube.R_Ny), dtype=bool)
        )
        qy0 = np.ma.array(
            data=qy0, mask=np.zeros((datacube.R_Nx, datacube.R_Ny), dtype=bool)
        )
        for (rx, ry) in tqdmnd(
            datacube.R_Nx,
            datacube.R_Ny,
            desc="Finding origins",
            unit="DP",
            unit_scale=True,
        ):
            if mask[rx, ry]:
                dp = datacube.data[rx, ry, :, :]
                _qx0, _qy0 = np.unravel_index(
                    np.argmax(gaussian_filter(dp, r)), (datacube.Q_Nx, datacube.Q_Ny)
                )
                _mask = np.hypot(qxx - _qx0, qyy - _qy0) < r * rscale
                qx0.data[rx, ry], qy0.data[rx, ry] = get_CoM(dp * _mask)
            else:
                qx0.mask, qy0.mask = True, True

    return qx0, qy0


def get_origin_from_braggpeaks(braggpeaks, Q_Nx, Q_Ny, findcenter="CoM", bvm=None):
    """
    Gets the diffraction shifts using detected Bragg disk positions.

    First, an guess at the unscattered beam position is determined, either by taking the
    CoM of the Bragg vector map, or by taking its maximal pixel.  If the CoM is used, an
    additional refinement step is used where we take the CoM of a Bragg vector map
    contructed from a first guess at the central Bragg peaks (as opposed to the BVM of all
    BPs). Once a unscattered beam position is determined, the Bragg peak closest to this
    position is identified. The shifts in these peaks positions from their average are
    returned as the diffraction shifts.

    Args:
        braggpeaks (PointListArray): the Bragg peak positions
        Q_Nx, Q_Ny (ints): the shape of diffration space
        findcenter (str): specifies the method for determining the unscattered beam
            position options: 'CoM', or 'max'
        bvm (array or None): the braggvector map. If None (default), the bvm is
            calculated

    Returns:
        (3-tuple): A 3-tuple comprised of:

            * **qx0** *((R_Nx,R_Ny)-shaped array)*: the origin x-coord
            * **qy0** *((R_Nx,R_Ny)-shaped array)*: the origin y-coord
            * **braggvectormap** *((R_Nx,R_Ny)-shaped array)*: the Bragg vector map of only
              the Bragg peaks identified with the unscattered beam. Useful for diagnostic
              purposes.
    """
    assert isinstance(braggpeaks, PointListArray), "braggpeaks must be a PointListArray"
    assert all([isinstance(item, (int, np.integer)) for item in [Q_Nx, Q_Ny]])
    assert isinstance(findcenter, str), "center must be a str"
    assert findcenter in ["CoM", "max"], "center must be either 'CoM' or 'max'"
    R_Nx, R_Ny = braggpeaks.shape

    # Get guess at position of unscattered beam
    if bvm is None:
        braggvectormap_all = get_bragg_vector_map(braggpeaks, Q_Nx, Q_Ny)
    else:
        braggvectormap_all = bvm
    if findcenter == "max":
        x0, y0 = np.unravel_index(
            np.argmax(gaussian_filter(braggvectormap_all, 10)), (Q_Nx, Q_Ny)
        )
    else:
        x0, y0 = get_CoM(braggvectormap_all)
        braggvectormap = np.zeros_like(braggvectormap_all)
        for Rx in range(R_Nx):
            for Ry in range(R_Ny):
                pointlist = braggpeaks.get_pointlist(Rx, Ry)
                if pointlist.length > 0:
                    r2 = (pointlist.data["qx"] - x0) ** 2 + (
                        pointlist.data["qy"] - y0
                    ) ** 2
                    index = np.argmin(r2)
                    braggvectormap = add_to_2D_array_from_floats(
                        braggvectormap,
                        pointlist.data["qx"][index],
                        pointlist.data["qy"][index],
                        pointlist.data["intensity"][index],
                    )
        x0, y0 = get_CoM(braggvectormap)

    # Get Bragg peak closest to unscattered beam at each scan position
    braggvectormap = np.zeros_like(braggvectormap_all)
    qx0 = np.zeros((R_Nx, R_Ny))
    qy0 = np.zeros((R_Nx, R_Ny))
    for Rx in range(R_Nx):
        for Ry in range(R_Ny):
            pointlist = braggpeaks.get_pointlist(Rx, Ry)
            if pointlist.length > 0:
                r2 = (pointlist.data["qx"] - x0) ** 2 + (pointlist.data["qy"] - y0) ** 2
                index = np.argmin(r2)
                braggvectormap = add_to_2D_array_from_floats(
                    braggvectormap,
                    pointlist.data["qx"][index],
                    pointlist.data["qy"][index],
                    pointlist.data["intensity"][index],
                )
                qx0[Rx, Ry] = pointlist.data["qx"][index]
                qy0[Rx, Ry] = pointlist.data["qy"][index]

    return qx0, qy0, braggvectormap

def get_origin_brightest_disk(
        datacube,
        probe_kernel,
        qxyInit = None,
        probe_mask_size=None,
        subpixel=None,
        upsample_factor=16,
        mask=None):    
    """
    Find the origin for all diffraction patterns in a datacube, by finding the
    brightest peak and then masking around that peak.

    Args:
        datacube (DataCube): the data
        probe_kernel (array): probe kernel for disk detection
        qxyInit (array or None): (qx0,qy0) origin for choosing the peak, or `None`.
            If `None`, the origin is the mean diffraction pattern is computed,
            which may be slow for large datasets, and is used to compute
            `(qx0,qy0)`.
        probe_mask_size (float): mask size in pixels. If set to None, we set it
            to 2*probe radius estimate.
        sub_pixel (str): 'None' or 'poly' or 'multicorr'
        upsample_factor (int): upsample factor
        mask (ndarray or None): if not None, should be an (Q_Nx,Q_Ny) shaped
            boolean array. Mask will be applied to diffraction patterns before
            finding the center.
        probe_mask_std_scale (float): size of Gaussian mask sigma. If set to
            None, function will estimate probe size.

    Returns:
        2 (R_Nx,R_Ny)-shaped ndarrays: the origin, (x,y) at each scan position
    """
    if probe_mask_size is None:
        probe_mask_size, px, py = get_probe_size(np.fft.fftshift(probe_kernel))
        probe_mask_size *= 2

    # take fft of probe kernel
    probe_kernel_FT = np.conj(np.fft.fft2(probe_kernel))

    if qxyInit is None:
        # find initial guess for probe center using mean diffraction pattern
        diff_mean = np.mean(datacube.data,axis=(0,1))
        peaks = _find_Bragg_disks_single_DP_FK(
            diff_mean,
            probe_kernel_FT,
            corrPower = 1,
            sigma = 0,
            edgeBoundary = 0,
            minRelativeIntensity = 0,
            minAbsoluteIntensity = 0,
            relativeToPeak = 0,
            maxNumPeaks = 1,
            subpixel = subpixel,
            upsample_factor = upsample_factor)
        qxyInit = np.array([peaks.data['qx'],peaks.data['qy']])

    # Create mask
    qx = np.arange(datacube.Q_Nx) - qxyInit[0]
    qy = np.arange(datacube.Q_Ny) - qxyInit[1]
    qya,qxa = np.meshgrid(qy,qx)
    if mask is None:
        mask = np.exp((qxa**2 + qya**2)/(-2*probe_mask_size**2))
    else:
        mask *= np.exp((qxa**2 + qya**2)/(-2*probe_mask_size**2))

    # init output arrays
    qx0_ar = np.zeros((datacube.R_Nx,datacube.R_Ny))
    qy0_ar = np.zeros((datacube.R_Nx,datacube.R_Ny))

    for (rx,ry) in tqdmnd(datacube.R_Nx,datacube.R_Ny,desc='Finding origins',unit='DP',unit_scale=True):
    # for (rx,ry) in tqdmnd(2,2,desc='Finding origins',unit='DP',unit_scale=True):
        if subpixel is None:
            dp_corr = np.real(np.fft.ifft2(np.fft.fft2(datacube.data[rx,ry,:,:] * mask) * probe_kernel_FT))
            ind2D = np.unravel_index(np.argmax(dp_corr), dp_corr.shape)
            qx0_ar[rx,ry] = ind2D[0]
            qy0_ar[rx,ry] = ind2D[1]
        else:
            peaks = _find_Bragg_disks_single_DP_FK(
                datacube.data[rx,ry,:,:] * mask,
                probe_kernel_FT,
                corrPower = 1,
                sigma = 0,
                edgeBoundary = 0,
                minRelativeIntensity = 0,
                minAbsoluteIntensity = 0,
                relativeToPeak = 0,
                maxNumPeaks = 1,
                subpixel = subpixel,
                upsample_factor = upsample_factor)
            qx0_ar[rx,ry] = peaks.data['qx']
            qy0_ar[rx,ry] = peaks.data['qy']

    return qx0_ar, qy0_ar

def get_origin_single_dp_beamstop(DP: np.ndarray,mask: np.ndarray):
    """
    Find the origin for a single diffraction pattern, assuming there is a beam stop.

    Args:
        DP (np array): diffraction pattern
        mask (np array): boolean mask which is False under the beamstop and True
            in the diffraction pattern. One approach to generating this mask
            is to apply a suitable threshold on the average diffraction pattern
            and use binary opening/closing to remove and holes

    Returns:
        qx0, qy0 (tuple) measured center position of diffraction pattern
    """

    imCorr = np.real(
        np.fft.ifft2(
            np.fft.fft2(DP * mask)
            * np.conj(np.fft.fft2(np.rot90(DP, 2) * np.rot90(mask, 2)))
        )
    )

    xp, yp = np.unravel_index(np.argmax(imCorr), imCorr.shape)

    dx = ((xp + DP.shape[0] / 2) % DP.shape[0]) - DP.shape[0] / 2
    dy = ((yp + DP.shape[1] / 2) % DP.shape[1]) - DP.shape[1] / 2

    return (DP.shape[0] + dx) / 2, (DP.shape[1] + dy) / 2


def get_origin_beamstop(datacube: DataCube, mask: np.ndarray):
    """
    Find the origin for each diffraction pattern, assuming there is a beam stop.

    Args:
        datacube (DataCube)
        mask (np array): boolean mask which is False under the beamstop and True
            in the diffraction pattern. One approach to generating this mask
            is to apply a suitable threshold on the average diffraction pattern
            and use binary opening/closing to remove any holes

    Returns:
        qx0, qy0 (tuple of np arrays) measured center position of each diffraction pattern
    """

    qx0 = np.zeros(datacube.data.shape[:2])
    qy0 = np.zeros_like(qx0)

    for rx, ry in tqdmnd(datacube.R_Nx, datacube.R_Ny):
        x, y = get_origin_single_dp_beamstop(datacube.data[rx, ry, :, :], mask)

        qx0[rx,ry] = x
        qy0[rx,ry] = y

    return qx0, qy0

def get_origin_beamstop_braggpeaks(braggpeaks,center_guess,radii,Q_Nx,Q_Ny,
                                   max_dist=2,max_iter=1):
    """
    Find the origin from a set of braggpeaks assuming there is a beamstop, by identifying
    pairs of conjugate peaks inside an annular region and finding their centers of mass.

    Args:
        braggpeaks (PointListArray):
        center_guess (2-tuple): qx0,qy0
        radii (2-tuple): the inner and outer radii of the specified annular region
        Q_Nx,Q_Ny: the shape of diffraction space
        max_dist (number): the maximum allowed distance between the reflection of two
            peaks to consider them conjugate pairs
        max_iter (integer): for values >1, repeats the algorithm after updating center_guess

    Returns:
        (2d masked array): the origins
    """
    assert(isinstance(braggpeaks,PointListArray))
    R_Nx,R_Ny = braggpeaks.shape

    # remove peaks outside the annulus
    braggpeaks_masked = braggpeaks.copy()
    for rx in range(R_Nx):
        for ry in range(R_Ny):
            pl = braggpeaks_masked.get_pointlist(rx,ry)
            qr = np.hypot(pl.data['qx']-center_guess[0],
                          pl.data['qy']-center_guess[1])
            rm = np.logical_not(np.logical_and(qr>=radii[0],qr<=radii[1]))
            pl.remove_points(rm)

    # Find all matching conjugate pairs of peaks
    center_curr = center_guess
    for ii in range(max_iter):
        centers = np.zeros((R_Nx,R_Ny,2))
        found_center = np.zeros((R_Nx,R_Ny),dtype=bool)
        for rx in range(R_Nx):
            for ry in range(R_Ny):

                # Get data
                pl = braggpeaks_masked.get_pointlist(rx,ry)
                is_paired = np.zeros(len(pl.data),dtype=bool)
                matches = []

                # Find matching pairs
                for i in range(len(pl.data)):
                    if not is_paired[i]:
                        x,y = pl.data['qx'][i],pl.data['qy'][i]
                        x_r = -x+2*center_curr[0]
                        y_r = -y+2*center_curr[1]
                        dists = np.hypot(x_r-pl.data['qx'],y_r-pl.data['qy'])
                        dists[is_paired] = 2*max_dist
                        matched = dists<=max_dist
                        if(any(matched)):
                            match = np.argmin(dists)
                            matches.append((i,match))
                            is_paired[i],is_paired[match] = True,True

                # Find the center
                if len(matches)>0:
                    x0,y0 = [],[]
                    for i in range(len(matches)):
                        x0.append(np.mean(pl.data['qx'][list(matches[i])]))
                        y0.append(np.mean(pl.data['qy'][list(matches[i])]))
                    x0,y0 = np.mean(x0),np.mean(y0)
                    centers[rx,ry,:] = x0,y0
                    found_center[rx,ry] = True
                else:
                    found_center[rx,ry] = False

        # Update current center guess
        x0_curr = np.mean(centers[found_center,0])
        y0_curr = np.mean(centers[found_center,1])
        center_curr = x0_curr,y0_curr

    # return
    found_center = np.logical_not(np.dstack([found_center,found_center]))
    origins = np.ma.array(data=centers, mask=found_center)
    return origins




### Functions for fitting the origin


def fit_origin(
    qx0_meas,
    qy0_meas,
    mask=None,
    fitfunction="plane",
    returnfitp=False,
    robust=False,
    robust_steps=3,
    robust_thresh=2,
):
    """
    Fits the position of the origin of diffraction space to a plane or parabola,
    given some 2D arrays (qx0_meas,qy0_meas) of measured center positions, optionally
    masked by the Boolean array `mask`.

    Args:
        qx0_meas (2d array): measured origin x-position
        qy0_meas (2d array): measured origin y-position
        mask (2b boolean array, optional): ignore points where mask=True
        fitfunction (str, optional): must be 'plane' or 'parabola' or 'bezier_two'
        returnfitp (bool, optional): if True, returns the fit parameters
        robust (bool, optional): If set to True, fit will be repeated with outliers
            removed.
        robust_steps (int, optional): Optional parameter. Number of robust iterations
                                performed after initial fit.
        robust_thresh (int, optional): Threshold for including points, in units of
            root-mean-square (standard deviations) error of the predicted values after
            fitting.

    Returns:
        (variable): Return value depends on returnfitp. If ``returnfitp==False``
        (default), returns a 4-tuple containing:

            * **qx0_fit**: *(ndarray)* the fit origin x-position
            * **qy0_fit**: *(ndarray)* the fit origin y-position
            * **qx0_residuals**: *(ndarray)* the x-position fit residuals
            * **qy0_residuals**: *(ndarray)* the y-position fit residuals

        If ``returnfitp==True``, returns a 2-tuple.  The first element is the 4-tuple
        described above.  The second element is a 4-tuple (popt_x,popt_y,pcov_x,pcov_y)
        giving fit parameters and covariance matrices with respect to the chosen
        fitting function.
    """
    assert isinstance(qx0_meas, np.ndarray) and len(qx0_meas.shape) == 2
    assert isinstance(qx0_meas, np.ndarray) and len(qy0_meas.shape) == 2
    assert qx0_meas.shape == qy0_meas.shape
    assert mask is None or mask.shape == qx0_meas.shape and mask.dtype == bool
    assert fitfunction in ("plane", "parabola", "bezier_two")
    if fitfunction == "plane":
        f = plane
    elif fitfunction == "parabola":
        f = parabola
    elif fitfunction == "bezier_two":
        f = bezier_two
    else:
        raise Exception("Invalid fitfunction '{}'".format(fitfunction))

    # Fit data
    if mask is None:
        popt_x, pcov_x, qx0_fit = fit_2D(
            f,
            qx0_meas,
            robust=robust,
            robust_steps=robust_steps,
            robust_thresh=robust_thresh,
        )
        popt_y, pcov_y, qy0_fit = fit_2D(
            f,
            qy0_meas,
            robust=robust,
            robust_steps=robust_steps,
            robust_thresh=robust_thresh,
        )

    else:
        popt_x, pcov_x, qx0_fit = fit_2D(
            f,
            qx0_meas,
            robust=robust,
            robust_steps=robust_steps,
            robust_thresh=robust_thresh,
            data_mask=mask == False,
        )
        popt_y, pcov_y, qy0_fit = fit_2D(
            f,
            qy0_meas,
            robust=robust,
            robust_steps=robust_steps,
            robust_thresh=robust_thresh,
            data_mask=mask == False,
        )

    # Compute residuals
    qx0_residuals = qx0_meas - qx0_fit
    qy0_residuals = qy0_meas - qy0_fit

    # Return
    if not returnfitp:
        return qx0_fit, qy0_fit, qx0_residuals, qy0_residuals
    else:
        return (qx0_fit, qy0_fit, qx0_residuals, qy0_residuals), (
            popt_x,
            popt_y,
            pcov_x,
            pcov_y,
        )


### Older / soon-to-be-deprecated functions for finding the origin


def find_outlier_shifts(xshifts, yshifts, n_sigma=10, edge_boundary=0):
    """
    Finds outliers in the shift matrices.

    Gets a score function for each scan position Rx,Ry, given by the sum of the absolute values of
    the difference between the shifts at this position and all 8 NNs. Calculates a histogram of the
    scoring function, fits a gaussian to its initial peak, and sets a cutoff value to n_sigma times
    its standard deviation. Values beyond this cutoff are deemed outliers, as are scan positions
    within edge_boundary pixels of the edge of real space.

    Accepts:
        xshifts         ((R_Nx,R_Ny)-shaped array) the shifts in x
        yshifts         ((R_Nx,R_Ny)-shaped array) the shifts in y
        n_sigma         (float) the cutoff value for the score function, in number of std
        edge_boundary   (int) number of pixels near the mask edge to mark as outliers

    Returns:
        mask            ((R_nx,R_ny)-shaped array of bools) the outlier mask
        score           ((R_nx,R_ny)-shaped array) the outlier scores
        cutoff          (float) the score cutoff value
    """
    # Get score
    score = np.zeros_like(xshifts)
    score[:-1, :] += np.abs(
        xshifts[:-1, :] - np.roll(xshifts, (-1, 0), axis=(0, 1))[:-1, :]
    )
    score[1:, :] += np.abs(
        xshifts[1:, :] - np.roll(xshifts, (1, 0), axis=(0, 1))[1:, :]
    )
    score[:, :-1] += np.abs(
        xshifts[:, :-1] - np.roll(xshifts, (0, -1), axis=(0, 1))[:, :-1]
    )
    score[:, 1:] += np.abs(
        xshifts[:, 1:] - np.roll(xshifts, (0, 1), axis=(0, 1))[:, 1:]
    )
    score[:-1, :-1] += np.abs(
        xshifts[:-1, :-1] - np.roll(xshifts, (-1, -1), axis=(0, 1))[:-1, :-1]
    )
    score[1:, :-1] += np.abs(
        xshifts[1:, :-1] - np.roll(xshifts, (1, -1), axis=(0, 1))[1:, :-1]
    )
    score[:-1, 1:] += np.abs(
        xshifts[:-1, 1:] - np.roll(xshifts, (-1, 1), axis=(0, 1))[:-1, 1:]
    )
    score[1:, 1:] += np.abs(
        xshifts[1:, 1:] - np.roll(xshifts, (1, 1), axis=(0, 1))[1:, 1:]
    )
    score[:-1, :] += np.abs(
        yshifts[:-1, :] - np.roll(yshifts, (-1, 0), axis=(0, 1))[:-1, :]
    )
    score[1:, :] += np.abs(
        yshifts[1:, :] - np.roll(yshifts, (1, 0), axis=(0, 1))[1:, :]
    )
    score[:, :-1] += np.abs(
        yshifts[:, :-1] - np.roll(yshifts, (0, -1), axis=(0, 1))[:, :-1]
    )
    score[:, 1:] += np.abs(
        yshifts[:, 1:] - np.roll(yshifts, (0, 1), axis=(0, 1))[:, 1:]
    )
    score[:-1, :-1] += np.abs(
        yshifts[:-1, :-1] - np.roll(yshifts, (-1, -1), axis=(0, 1))[:-1, :-1]
    )
    score[1:, :-1] += np.abs(
        yshifts[1:, :-1] - np.roll(yshifts, (1, -1), axis=(0, 1))[1:, :-1]
    )
    score[:-1, 1:] += np.abs(
        yshifts[:-1, 1:] - np.roll(yshifts, (-1, 1), axis=(0, 1))[:-1, 1:]
    )
    score[1:, 1:] += np.abs(
        yshifts[1:, 1:] - np.roll(yshifts, (1, 1), axis=(0, 1))[1:, 1:]
    )
    score[1:-1, 1:-1] /= 8.0
    score[:1, 1:-1] /= 5.0
    score[-1:, 1:-1] /= 5.0
    score[1:-1, :1] /= 5.0
    score[1:-1, -1:] /= 5.0
    score[0, 0] /= 3.0
    score[0, -1] /= 3.0
    score[-1, 0] /= 3.0
    score[-1, -1] /= 3.0

    # Get mask and return
    cutoff = np.std(score) * n_sigma
    mask = score > cutoff
    if edge_boundary > 0:
        mask[:edge_boundary, :] = True
        mask[-edge_boundary:, :] = True
        mask[:, :edge_boundary] = True
        mask[:, -edge_boundary:] = True

    return mask, score, cutoff


def center_braggpeaks(braggpeaks, qx0=None, qy0=None, coords=None, name=None):
    """
    Shift the braggpeaks positions to center them about the origin, given
    either by (qx0,qy0) or by the Coordinates instance coords. Either
    (qx0,qy0) or coords must be specified.

    Accepts:
        braggpeaks  (PointListArray) the detected, unshifted bragg peaks
        qx0,qy0     ((R_Nx,R_Ny)-shaped arrays) the position of the origin,
                    or scalar values for constant origin position.
        coords      (Coordinates) an object containing the origin positions
        name        (str, optional) a name for the returned PointListArray.
                    If unspecified, takes the old PLA name, removes '_raw'
                    if present at the end of the string, then appends
                    '_centered'.

    Returns:
        braggpeaks_centered  (PointListArray) the centered Bragg peaks
    """
    assert isinstance(braggpeaks, PointListArray)
    assert (qx0 is not None and qy0 is not None) != (
        coords is not None
    ), "Either (qx0,qy0) or coords must be specified"
    if coords is not None:
        qx0, qy0 = coords.get_origin()
        assert (
            qx0 is not None and qy0 is not None
        ), "coords did not contain center position"
    if name is None:
        sl = braggpeaks.name.split("_")
        _name = "_".join(
            [s for i, s in enumerate(sl) if not (s == "raw" and i == len(sl) - 1)]
        )
        name = _name + "_centered"
    assert isinstance(name, str)
    braggpeaks_centered = braggpeaks.copy(name=name)

    if np.isscalar(qx0) & np.isscalar(qy0):
        for Rx in range(braggpeaks_centered.shape[0]):
            for Ry in range(braggpeaks_centered.shape[1]):
                pointlist = braggpeaks_centered.get_pointlist(Rx, Ry)
                pointlist.data["qx"] -= qx0
                pointlist.data["qy"] -= qy0
    else:
        for Rx in range(braggpeaks_centered.shape[0]):
            for Ry in range(braggpeaks_centered.shape[1]):
                pointlist = braggpeaks_centered.get_pointlist(Rx, Ry)
                qx, qy = qx0[Rx, Ry], qy0[Rx, Ry]
                pointlist.data["qx"] -= qx
                pointlist.data["qy"] -= qy

    return braggpeaks_centered
