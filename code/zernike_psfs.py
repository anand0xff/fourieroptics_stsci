#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Read in Zernike cube and explore aberrated psf morphology at 2x Nyquist image sampling
"""


### Libraries

import sys
import astropy.io.fits as fits
import numpy as np
import scipy.ndimage.interpolation
import poppy.matrixDFT as matrixDFT

stddevs = (0.3, 1.0, 3.0) # in radians - aberration std devs.
nab = len(stddevs)

zerns = fits.getdata("ZernikeFitting/ZfunctionsUnitVar.fits") # 100 x 100 arrays in cube
print("nzerns, nx, ny =", zerns.shape)
gamma = 4 # Soummer gamma of oversampling in the Fourier domain. Use integer.
imagefov = zerns.shape[1]//gamma  # in ft's reselts - lam/D if an image plane
npix = gamma*imagefov

# for storing nab eg 3 psfs per zernike, varying strenghts and the zernike function
psfs = np.zeros( (zerns.shape[0], npix, npix*(nab+1)) ) # gamma oversampling

ft = matrixDFT.MatrixFourierTransform()

pupil = zerns[0,:,:].copy()
# perfect image: 
imagefield = ft.perform(pupil, imagefov, gamma*imagefov)
imageintensity =  (imagefield*imagefield.conj()).real
perfectpeakintensity = imageintensity.max()

fits.writeto('perfectPSF.fits', imageintensity/perfectpeakintensity, overwrite=True, checksum=True)

for nz in range(zerns.shape[0]):
    for iab, ab in enumerate(stddevs):
        imagefield = ft.perform(pupil*np.exp(1j*zerns[nz,:,:]*ab), imagefov, npix)
        imageintensity =  (imagefield*imagefield.conj()).real
        psfs[nz, :, iab*npix:(iab+1)*npix] =  imageintensity/perfectpeakintensity
        #sfs[nz, :, (iab+1)*npix:] = scipy.ndimage.interpolation.zoom(zerns[nz,:,:], 1.0/gamma,
        #                                output=None, order=0)
    displayzern = zerns[nz,:,:] - zerns[nz,:,:].min() 
    # for all nonpiston Z's...
    if nz != 0: displayzern = (zerns[nz,:,:] - zerns[nz,:,:].min()) /  (zerns[nz,:,:].max() - zerns[nz,:,:].min())
    psfs[nz, :, nab*npix:] = displayzern * 0.5 # looks better w lower peak...

fits.writeto('zernedPSFcube.fits', psfs.astype(np.float32), overwrite=True)
