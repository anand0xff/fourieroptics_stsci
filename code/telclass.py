#! /usr/bin/env python
""" 
	anand@stsci.edu 2018

"""

import sys, os, time
import numpy as np
import scipy
from astropy.io import fits
import poppy.matrixDFT as matrixDFT
import utils
""" matrixDFT.perform():
    Parameters
    ----------
    plane : 2D ndarray
        2D array (either real or complex) representing the input image plane or
        pupil plane to transform.
    nlamD : float or 2-tuple of floats (nlamDY, nlamDX)
        Size of desired output region in lambda / D units, assuming that the
        pupil fills the input array (corresponds to 'm' in
        Soummer et al. 2007 4.2). This is in units of the spatial frequency that
        is just Nyquist sampled by the input array.) If given as a tuple,
        interpreted as (nlamDY, nlamDX).
    npix : int or 2-tuple of ints (npixY, npixX)
        Number of pixels per side side of destination plane array (corresponds
        to 'N_B' in Soummer et al. 2007 4.2). This will be the # of pixels in
        the image plane for a forward transformation, in the pupil plane for an
        inverse. If given as a tuple, interpreted as (npixY, npixX)."""

def exer1(odir):
    """ Let's get something for nothing if we can!!! """

    # instantiate an mft object:
    ft = matrixDFT.MatrixFourierTransform()

    npup = 100
    radius = 20.0
    fp_size_reselts = (100, 200, 300, 400)
    fp_npixels = (100, 200, 300, 400)  
    pupil = utils.makedisk(npup, radius=radius)
    fits.PrimaryHDU(pupil).writeto(odir+"/ex1_pupil.fits", overwrite=True) # write pupil file
    for (fpr,fpnpix) in zip(fp_npixels,fp_size_reselts):
        imagearray = np.zeros((400,400)) # create same-sized array for all 4 FOV's we calculate.
        imagefield = ft.perform(pupil, fpr, fpnpix)
        imageintensity = (imagefield*imagefield.conj()).real
        psf = imageintensity / imageintensity.max()  # normalize to peak intensity unity
        imagearray[200-fpnpix//2:200+fpnpix//2,200-fpnpix//2:200+fpnpix//2] = psf # center image in largest array size
        fits.PrimaryHDU(imagearray).writeto(odir+"/ex1_nfppix{}_fpsize{}.fits".format(fpnpix,fpr), overwrite=True)

def exer2(odir):
    """ Are you simulating samples or pixels?  Image plane sampling effect """
    # instantiate an mft object:
    ft = matrixDFT.MatrixFourierTransform()

    npup = 100
    radius = 20.0
    fp_size_reselts = 10
    fp_npixels = (4, 8, 16, 32)  
    pupil = utils.makedisk(npup, radius=radius)
    fits.PrimaryHDU(pupil).writeto(odir+"/ex2_pupil.fits", overwrite=True) # write pupil file
    for fpnpix in fp_npixels:
        imagefield = ft.perform(pupil, fp_size_reselts, fpnpix)
        imageintensity = (imagefield*imagefield.conj()).real
        psf = imageintensity / imageintensity.max()  # normalize to peak intensity unity
        zpsf = scipy.ndimage.zoom(psf, 32//fpnpix, order=0)
        fits.PrimaryHDU(zpsf).writeto(odir+"/ex2_nfppix{}_zoom{}.fits".format(fpnpix,32//fpnpix), overwrite=True)

def exer3(odir):
    """ What do phase slopes - tilts - in the pupil plane do? """
    # instantiate an mft object:
    ft = matrixDFT.MatrixFourierTransform()
    npup = 100
    radius = 20.0
    fp_size_reselt = 100
    pupil = utils.makedisk(npup, radius=radius)
    tilts = ((0,0),  (0.3,0), (1.0,0), (3.0,0))
    phases = utils.phasearrays(npup, tilts)
    for nt, ph in enumerate(phases):
        pupilarray = pupil * np.exp(1j * ph)
        imagefield = ft.perform(pupilarray, fp_size_reselt, npup)
        image_intensity = (imagefield*imagefield.conj()).real
        psf = image_intensity / image_intensity.max()  # peak intensity unity
        fits.PrimaryHDU(psf).writeto( 
            odir+"/c2_ex3_tilt_a_{0:.3f}_b_{1:.3f}.fits".format(tilts[nt][0],tilts[nt][1]), 
            overwrite=True)

def exer4(odir):
    """ Precise control of image positioning using phase slopes? """
    # instantiate an mft object:
    ft = matrixDFT.MatrixFourierTransform()
    npup = 100
    radius = 20.0
    fp_size_reselt = 100
    pupil = utils.makedisk(npup, radius=radius)
    tilts = ((0,0),
             (0*np.pi/npup, 0), 
             (1*np.pi/npup, 0),
             (2*np.pi/npup, 0),
             (3*np.pi/npup, 0),
             (4*np.pi/npup, 0),
             (95*np.pi/npup, 0),
             (npup*np.pi/npup, 0))
    phases = utils.phasearrays(npup, tilts)
    for nt, ph in enumerate(phases):
        pupilarray = pupil * np.exp(1j * ph)
        imagefield = ft.perform(pupilarray, fp_size_reselt, npup)
        image_intensity = (imagefield*imagefield.conj()).real
        psf = image_intensity / image_intensity.max()  # peak intensity unity
        fits.PrimaryHDU(psf).writeto( 
            odir+"/c2_ex4_tiltPi_a_{0:.3f}_b_{1:.3f}.fits".format(tilts[nt][0],tilts[nt][1]), 
            overwrite=True)

def exer5(odir):
    """ Create ripples of phase aberration in the pupil
        to simulate the Ruffles Potato Chip Telescope (RPCT) """
    # instantiate an mft object:
    ft = matrixDFT.MatrixFourierTransform()
    npup = 100
    radius = 20.0
    fp_size_reselt = 100
    pupil = utils.makedisk(npup, radius=radius)
    number_of_waves_across = (2,3,4,5,6)

    peaks = (0.0, 0.1, 0.3, 1.0, 3.0) # radians, amplitude of phase ripple
    arrayshape = (npup, npup)
    diam_pupil = radius*2 # pixels
    center = (arrayshape[0]/2, arrayshape[1]/2)
    for nwaves in number_of_waves_across:
        for peak in peaks: 
            for offset in (0, ):
                for angle in (0, ):
                    spatialwavelen = diam_pupil / nwaves
                    offset = offset * np.pi/180.0
                    khat = np.array((np.sin(angle*np.pi/180.0), np.cos(angle*np.pi/180.0))) # unit vector
                    kwavedata = np.fromfunction(utils.kwave2d, arrayshape,
                            spatialwavelen=spatialwavelen,
                            center=center,
                            offset=offset,
                            khat=khat)
                    pupilarray = pupil * np.exp(1j * peak * kwavedata)
                    imagefield = ft.perform(pupilarray, fp_size_reselt, npup)
                    image_intensity = (imagefield*imagefield.conj()).real
                    psf = image_intensity / image_intensity.max()  # peak intensity unity
                    fits.PrimaryHDU(psf).writeto( 
                         odir+"/c2_ex5_phaseripple_{0:d}acrossD_peak_{1:.1f}.fits".format(nwaves,peak), 
                         overwrite=True)

def class1(tdir):
    # create output directory if it does not exist
    odir = tdir + '/c1'
    print("   class 1", odir)
    if not os.path.exists(odir):
        os.makedirs(odir)
    print("odir", odir)
    exer1(odir)
    exer2(odir)

def class2(tdir):
    # create output directory if it does not exist
    odir = tdir + '/c2'
    print("   class 2", odir)
    if not os.path.exists(odir):
        os.makedirs(odir)
    print("odir", odir)
    exer3(odir)
    exer4(odir)
    exer5(odir)

if __name__ == "__main__":
	
    # create output directory if it does not exist
    pathname = os.path.dirname(".")
    fullPath = os.path.abspath(pathname)
    topdir = fullPath + '/datadir'
    if not os.path.exists(topdir):
        os.makedirs(topdir)

    print("topdir", topdir)
    class1(topdir)
    #class2(topdir)



