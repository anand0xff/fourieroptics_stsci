#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import astropy.io.fits as fits
import numpy as np
import poppy
import poppy.matrixDFT as matrixDFT
import scipy.ndimage.interpolation


def write_zernike_cube(method="poppy"):
    """
    Function to write out a FITS file of a cube 36x100x400 where each slice is 3
    side-by-side PSFs with different standard deviations (0.3, 1, and 3) and a
    pupil image of the Zernicke used to create the PSFs

    "Explores aberrated PSF morphology at 2x Nyquist image sampling"

    Parameters
    ---------
    method : str
        Defines the method used to create the Zernikes. Either "poppy" or "original".
        "poppy" calls poppy.zernike.zernike1() function and uses Noll ordering.
        "original" is Anand's original code which reads in a previously created
        "ZfunctionsUnitVar.fits" file made from running ZernikeFitter.py.
    """

    stddevs = (0.3, 1.0, 3.0)  # in radians - aberration std devs
    nab = len(stddevs)

    if method.lower() == "original":
        zerns = fits.getdata("ZernikeFitting/ZfunctionsUnitVar.fits")  # 100 x 100 arrays in cube
    elif method.lower() == "poppy":
        zerns = np.array([poppy.zernike.zernike1(i, npix=100, outside=0) for i in range(1, 37)])
    else:
        raise TypeError("Not a valid method selection")

    print("nzerns, nx, ny =", zerns.shape)

    gamma = 4  # Soummer gamma of oversampling in the Fourier domain. Use integer.
    imagefov = zerns.shape[1]//gamma  # in ft's results - lam/D if an image plane
    npix = gamma*imagefov

    # For storing nab eg 3 PSFs per Zernike, varying strengths and the Zernike function
    psfs = np.zeros((zerns.shape[0], npix, npix*(nab+1)))  # gamma oversampling

    ft = matrixDFT.MatrixFourierTransform()

    pupil = zerns[0, :, :].copy()

    # Perfect image
    imagefield = ft.perform(pupil, imagefov, gamma*imagefov)
    imageintensity = (imagefield*imagefield.conj()).real
    perfectpeakintensity = imageintensity.max()

    fits.writeto('perfectPSF.fits', imageintensity/perfectpeakintensity, overwrite=True, checksum=True)

    for nz in range(zerns.shape[0]):
        for iab, ab in enumerate(stddevs):
            imagefield = ft.perform(pupil*np.exp(1j*zerns[nz, :, :]*ab), imagefov, npix)
            imageintensity = (imagefield*imagefield.conj()).real
            psfs[nz, :, iab*npix:(iab+1)*npix] = imageintensity / perfectpeakintensity
            # sfs[nz, :, (iab+1)*npix:] = scipy.ndimage.interpolation.zoom(zerns[nz,:,:], 1.0/gamma,
            #                                output=None, order=0)

        displayzern = zerns[nz, :, :] - zerns[nz, :, :].min()

        # For all non-piston Z's
        if nz != 0:
            displayzern = (zerns[nz, :, :] - zerns[nz, :, :].min()) / (zerns[nz, :, :].max() - zerns[nz, :, :].min())

        psfs[nz, :, nab*npix:] = displayzern * 0.5  # looks better w lower peak

    fits.writeto('zernedPSFcube_{}.fits'.format(method.lower()), psfs.astype(np.float32), overwrite=True)


if __name__ == "__main__":
    write_zernike_cube(method="poppy")
