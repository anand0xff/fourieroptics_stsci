#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created by Anand Sivaramakrishnan anand@stsci.edi 2018 12 28

This file is licensed under the Creative Commons Attribution-Share Alike
license versions 3.0 or higher, see
http://creativecommons.org/licenses/by-sa/3.0/

Python 3

    matrixDFT.perform():  (also matrixDFT.inverse():)
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
        inverse. If given as a tuple, interpreted as (npixY, npixX).
"""


### Libraries/modules

import os, sys
import astropy.io.fits as fits
import numpy as np
import poppy.matrixDFT as matrixDFT
import utils

PUPIL = "pupil"
IMAGE = "image"

# This will set up default printing of numpy to 3 dec places, scientific notation
#np.set_printoptions(precision=3, threshold=None, edgeitems=None, linewidth=None,
#                    suppress=None, nanstr=None, infstr=None, formatter={'float': '{: 0.3e}'.format} )

def create_input_datafiles(rfn=None):
    """
        Pupil and image data, true (zero mean) phase map (rad) 
        No de-tilting of the thase done.
        Didactic case, no input parameters: create the pupil & monochromatic image 
        on appropriate pixel scales.  
        Pupil and image arrays of same size, ready to FT into each 
        other losslessly.  For real data you'll need to worry about adjusting
        sizes to satify this sampling relation between the two planes.
        For finite bandwidth data image simulation will loop over wavelengths.
        For coarsely sampled pixels image simulation will need to use finer
        sampling and rebin to detector pixel sizes.
        anand@stsci.edu Jan 2019
    """
    pupil0 = utils.makedisk(250, radius=50) # D=100 pix, array 250 pix
    pupil1 = (pupil0 - utils.makedisk(250, radius=15, ctr=(125.0,75.0)))*pupil0
    pupil2 = (pupil1 - utils.makedisk(250, radius=15, ctr=(90.0,90.0)))*pupil0
    pupil3 = (pupil2 - utils.makedisk(250, radius=15, ctr=(75.0,125.0)))*pupil0
    fits.writeto(rfn+"0__input_pup.fits", pupil0, overwrite=True)
    fits.writeto(rfn+"1__input_pup.fits", pupil1, overwrite=True)
    fits.writeto(rfn+"2__input_pup.fits", pupil2, overwrite=True)
    fits.writeto(rfn+"3__input_pup.fits", pupil3, overwrite=True)


    for pnum in (0,1,2,3):
        pupil = fits.getdata(rfn+"{:1d}__input_pup.fits".format(pnum))
        offctrbump=(pupil.shape[0]/2.0+30,pupil.shape[1]/2.0+20.0)  # off-center bump
        ctrbump=(pupil.shape[0]/2.0,pupil.shape[1]/2.0)  # central bump
        #quarter wave phase bump, off center
        phase = (2.0*np.pi/4.0) * utils.makegauss(pupil.shape[0], 
                                                  ctr=offctrbump, 
                                                  sigma=10.0) * pupil
        phase = de_mean(phase, pupil) # zero mean phase - doesn't change image
        fits.writeto(rfn+"{:1d}__input_truepha.fits".format(pnum), phase, overwrite=True)
        mft = matrixDFT.MatrixFourierTransform()
        imagefield = mft.perform(pupil * np.exp(1j*phase), pupil.shape[0], pupil.shape[0])
        image =  (imagefield*imagefield.conj()).real 
        fits.writeto(rfn+"{:1d}__input_img.fits".format(pnum), image/image.sum(), overwrite=True)
        del mft

def data_input(pupfn=None, imgfn=None, truephasefn=None, rfn=None):
    """
        Get pupil and image data,
        No de-tilting of the thase done.
        Returns pupil and image arrays of samesize, ready to FT into each 
        other correctly.  For real data you'll need to worry about adjusting
        sizes to satify this sampling relation between the two planes.
        For finite bandwidth data image simulation will loop over wavelengths.
        For coarsely sampled pixels image simulation will need to use finer
        sampling and rebin to detector pixel sizes.
        anand@stsci.edu Jan 2019
    """
    pupil = fits.getdata(rfn+pupfn)
    image = fits.getdata(rfn+imgfn)
    truephase =fits.getdata(rfn+truephasefn)
    return pupil, image/image.sum(), truephase

def arraystats(phase):
    return "mean {:+.4e}  sigma {:+.4e}  max {:+.4e}  min {:+.4e} ".format(
            phase.mean(), phase.std(), phase.max(), phase.min())
    
def phase_changes(pupil, interior, phase_old, phase_new, gain):
    """
        pupil: binary pupil array (grey edges OK)
        interior: one pixel clear of pupil edge, 0/1 array
        phase_old, phase_new: phase arrays to compare after de-meaning, de-tilting
        gain: the damping factor applied to the correction
        returns b - a phase difference array, standard dev of this phase difference,
        and the damped next phase to use

        interior unused here because we did not measure tilts of wavefronts
    """
    dphase = (phase_new - phase_old) * pupil
    newphase = (phase_old + gain*dphase) * pupil # next iteration's pupil phase
    newphase = de_mean(newphase, pupil)

    return dphase, dphase[np.where(np.greater(pupil,0))].std(), newphase

def de_mean(a, pupil):
    """ 
        remove the mean where there is live pupil (0/1)
    """
    return (a - a[np.where(pupil==1)].mean()) * pupil
    
def meantilt(a, support):
    """
        a: array like an OPD which you want to measure the mean tilt of.
        support_idx: indices array of region over which to take the mean of the tilts
        returns: (2-tuple) the mean tilt over the index array, in ["a" units] per pixel  
        TBD: for really accurate work you should take the pixels either side and 
        calculate the 'unbiased' tilt over the interior.

        In this 2018 version I changed tilt signs so higher value of array at
        higher index is positive tilt
        Remove tilt with 
        phasea[pupil_idx] -= ta[0]*pupil_idx[0] - ta[1] * pupil_idx[1] ?? (untested)
    """
    support_idx = np.where(support==1)
    xt,yt  = (np.zeros(a.shape), np.zeros(a.shape))
    xt[:-1,:] = a[1:,:] - a[:-1,:] 
    yt[:,:-1] = a[:,1:] - a[:,:-1] 
    print("\tMean tilt vector = {:.4e}, {:.4e}".format(xt[support_idx].mean(), yt[support_idx].mean()))
    return xt[support_idx].mean(), yt[support_idx].mean() # pure python x,y

def interior(pup):
    """ The interior of a pupil, one pixel either side is clear of the pupil edge
        Used to calculate average tilt over the pupil
        Only works for unapodized (grey edge pixels OK)
            mA,mB = meantilt(opd, np.where(insideNRM()==1))
            print "mA,mB  %8.1e  %8.1e  rad/pix  //  " % (mA, mB), 
    """
    support = np.where(np.equal(pup, 1), 1, 0)
    support[1:,:] =   support[1:,:]  * support[:-1, :]  #clip_loY ds9
    support[:-1,:] =  support[:-1,:] * support[1:, :]   #clip_hiY ds9
    support[:,1:] =   support[:,1:]  * support[:,:-1]   #clip_loX ds9
    support[1:,:] =   support[1:,:]  * support[:-1, :]  #clip_loY ds9
    return support.astype(np.uint8)


def powerfunc(carr):
    return ((carr*carr.conj()).real).sum()


class GerchbergSaxton:
    """
    Does a Gerchberg Saxton in focus phase retrieval example case without noise
    Uses lossless transforms both ways (like an fft but a dft is used for didactic clarity)

    Usage:

        import GerchbergSaxton as GS

        # create or obtain pupil with obstructions, apodizations
        pupilfn = None # creates pupil
        initphase = None # First guess at pupil phase array (radians wavefront) or None
        damp=0.3, # pupil phase correction  damping, dimensionless, <= 1 # UNUSED for clean didactc demonstration

        # read in or obtain intensity data from in-focus image
        # this data is on the sama angular scale as a lossless dft 
        # of the pupil array, for simplicity.  Trim it appropriately
        # ahead of time if necessary.
        imagefn = None # creates image with a gaussian phase bump aber

        # Save the first 5 iterations, then every 10th iteration
        # eg 1 2 3 4 5 10 20...
        history = (5,10) 

        # rms phase change threshold for convergence (radians wavefront)
        threshold = 0.1 # context:  Marechal approx gives Strehl 99% for phase rms 0.1

        # max number of iterations (phase, image, back to next phase)
        maxiter = 30

        # output file(s) root name
        fileroot = "GSdir/gstest"
        input files names will be appended to the fileroot

        # read in or obtain first guess of the wavefront phase in pupil
        gs = GS.GerchbergSaxton(pupilfn=pupilfn, 
                                imagefn=imagefn,
                                truephasefn=truephasefn,
                                outputtag="2", # kluge to match the numbered files for 4 didactic cases...
                                initphase=phase,  # initial guess, rad
                                damp=damp, # pupil phase iteration damping, dimensionless, < 1 or None (uses 1)
                                history=history, # record of which iterations you want to save for inspection later
                                threshold=threshold, # stop iterating when wavefront rms <= this, rad
                                maxiter=maxiter, # stop iterating after these number of iterations
                                fileroot=fileroot, # file name root for history
                                verbose=True,
                                )
        gs.gsloop()


### gerchbergsaxton.py files
| File name for Example 1   | Description
|--------------------------:|-------------------------------------------------------------------------|
| gs1__input_img.fits       | Image intensity file (unit power)
| gs1__input_pup.fits       | Pupil constraint file 
| gs1__input_truepha.fits   | Phase used to create image
| gs1__errorint.fits        | Input image - final image
| gs1__errorpha.fits        | Input phase - measured phase
| gs1_imgint.fits           | Cube of image intensities
| gs1_pupint.fits           | Cube of pupil intensities
| gs1_puppha.fits           | Cube of pupil phases

    """

    def __init__(self, pupilfn=None, imagefn=None, truephasefn=None, outputtag=0,
                 initphase=None,
                 damp=None, history=None, threshold=0.1,  # Strehl 99% ~ 1 - threshold^2
                 maxiter=None, fileroot=None, verbose=True):
        """
        Assumes square arrays, equal sized arrays for pupil, image
        Iterations are pupil - to image - to pupil
        """

        # Requested attributes:
        self.damp = damp
        self.history = history
        self.threshold = threshold
        self.maxiter = maxiter
        self.fileroot = fileroot
        self.verbose = verbose
        self.pupil, self.image, self.truephase = data_input(pupilfn, imagefn,  truephasefn, self.fileroot)
        self.tag = outputtag

        # Now for derived attributes:
        if initphase is None:
            self.initphase = self.pupil*0.0 # initial phase guess
        else:
            self.initphase = initphase
        self.phase = self.initphase.copy()
        self.pupilpower = (self.pupil*self.pupil).sum()
        self.mft = matrixDFT.MatrixFourierTransform()
        self.npix = self.pupil.shape[0] # size of the arrays
        # create interior of an unapodized pupil, for calculating mean tilts
        self.interior = interior(self.pupil)

        # Create initial guess at wavefront
        self.plane = PUPIL
        self.wavefront = self.pupil * np.exp(1j * self.phase)

        # List of phases to track evolution of GS
        self.iter = 0 # running count
        self.savediters = [-1,] # 
        self.pupilintensities = [self.pupil,] # 
        self.pupilphases = [self.initphase,] # 
        self.imageintensities = [self.image,] # 

        print("\nmaxiter {:d}  convergence if sigma < {:.2e}\n".format(self.maxiter, self.threshold))
        print("\nExample {:s} beginning\n".format(self.tag))

    def apply_pupil_constraint(self, ):
        """
        Replace absolute value of wavefront by pupil values, preserve the current phase
        """
        if self.plane == PUPIL:
            # this new phase estimate comes from the image plane tranformed to this pupil
            nextphase = np.angle(self.wavefront) * self.pupil

            # check to see how the guess at the wavefront is changing...
            if self.verbose: fits.writeto(self.fileroot+self.tag+"__interior.fits", self.interior, overwrite=True)
            if self.verbose: fits.writeto(self.fileroot+self.tag+"__nextphase.fits", nextphase, overwrite=True)
            self.phasedelta, phasedeltarms, dampednextphase = phase_changes(self.pupil, self.interior, 
                                                                      self.phase, nextphase, self.damp)
            print("\t{:3d}: Delta Phase stats: {:s} ".format(self.iter, 
                                                       arraystats(self.phasedelta[np.where(self.pupil==1)])))

            # update wavefront with the new phase estimate
            self.wavefront = self.pupil * np.exp(1j*dampednextphase)
            self.phase = dampednextphase
            return phasedeltarms
        else:
            sys.exit(" gerchbergsaxton.apply_pupil_constraint(): being used in the wrong plane")

    def apply_image_constraint(self, ):
        """
        Replace absolute value of wavefront by sqrt(image), preserve the current phase
        """
        if self.plane == IMAGE:
            self.imagephase = np.angle(self.wavefront)
            self.wavefront = np.sqrt(self.image) * np.exp(1j*self.imagephase)
            if self.verbose: print("\timage power {:.4e}".format(powerfunc(self.wavefront)))
            return None 
        else:
            sys.exit(" gerchbergsaxton.apply_image_constraint(): being used in the wrong plane")

    def propagate_wavefront(self, ):
        """
        Apply the appropriate transform to go to the next conjugate plane.
        """
        if self.plane == PUPIL:
            if self.verbose: print("  in pupil, going to image ")
            self.wavefront = self.mft.perform(self.wavefront, self.npix, self.npix)
            self.wavefront_image = self.wavefront
            self.plane = IMAGE
        elif self.plane == IMAGE:
            if self.verbose: print("  in image, going to pupil ")
            wavefront = self.mft.inverse(self.wavefront, self.npix, self.npix)
            power = (wavefront * wavefront.conj()).real.sum()
            self.wavefront = wavefront/np.sqrt(power) # preserve input total pupil power
            self.wavefront_pupil = self.wavefront
            self.plane = PUPIL
        else:
            sys.exit("GerchbergSaxton.propagate_wavefront() is confused: Neither pupil nor image plane")
        return None

    def iterate(self):
        """ iterate once, pupil to image then back to pupil,
            save data for the record if needed, 
            then apply pupil constraint again
        """
        if self.verbose: print("\niter:{:3d}".format(self.iter))
        self.propagate_wavefront() # pupil to image
        self.save_iteration()

        self.apply_image_constraint()

        self.propagate_wavefront() # image to pupil
        self.save_iteration()

        self.iter += 1

        rmsphasechange = self.apply_pupil_constraint()
        return rmsphasechange

    def save_iteration(self):
        """ keep a record of phases, intensities, etc.  
            pupil phases whole plane incl. outside pupil support
            pupil phases only on pupil support
            image intensities whole plane
        """
        if self.iter in range(history[0]) or self.iter%history[1]==0:
            if self.plane == PUPIL:
                pupilintensity = (self.wavefront*self.wavefront.conj()).real
                pupilintensity = pupilintensity/pupilintensity.sum()*self.pupilpower # preserve input total pupil power
                                                                                     # should we instead preserve power over pupil??
                                                                                     # near convergence these two should be 
                                                                                     # very close towards the end.  Just saying.
                                                                                     # But this is a prob density function-like
                                                                                     # array, so preserving total power is the
                                                                                     # commonsense thing to do.
                self.pupilintensities.append(pupilintensity) 
                self.pupilphases.append(self.pupil*np.angle(self.wavefront))
            if self.plane == IMAGE:
                imageintensity = (self.wavefront*self.wavefront.conj()).real
                self.imageintensities.append(imageintensity/imageintensity.sum()) # preserve unit image power
            self.savediters.append(self.iter)

    def gswrite(self):
        """ save to disk for inspection """

            
        # add the final iteration planes which may (or may not) be saved: 
        # TBD - fancy bookkeeping to not save this if the GS loop ended on a 'save_iteration' cycle.
        # Right now there's a small chance that the last-but-one and last slices will be identical.
        # Not a big deal - they should be pretty close anyway as we converge.s
        # final pupil intensity added to list
        pupilintensity = (self.wavefront_pupil*self.wavefront_pupil.conj()).real
        pupilintensity = pupilintensity/pupilintensity.sum()*self.pupilpower
        self.pupilintensities.append(pupilintensity)
        # final pupil phase added to list
        self.pupilphases.append(np.angle(self.wavefront_pupil)*self.pupil)
        # final image intensity added to list
        imageintensity = (self.wavefront_image*self.wavefront_image.conj()).real
        self.imageintensities.append(imageintensity/imageintensity.sum()) # preserve unit image power

        # Now save the sequence of iterations...
        fits.writeto(self.fileroot+self.tag+"_pupint.fits", np.array(tuple(self.pupilintensities)), overwrite=True)
        fits.writeto(self.fileroot+self.tag+"_puppha.fits", np.array(tuple(self.pupilphases)), overwrite=True)
        fits.writeto(self.fileroot+self.tag+"_imgint.fits", np.array(tuple(self.imageintensities)), overwrite=True)

        # Errors compared to input values:
        phase_meas_error = de_mean(self.truephase - self.pupilphases[-1], self.pupil)
        img_meas_error = self.image - self.imageintensities[-1]

        print("       Init Phase stats:  {:s} ".format(arraystats(self.truephase[np.where(self.pupil==1)])))
        print("       Final Phase stats: {:s} ".format(arraystats(self.pupilphases[-1][np.where(self.pupil==1)])))
        print("")
        print("Init - Final Phase stats: {:s} ".format(arraystats(phase_meas_error[np.where(self.pupil==1)])))
        print("       Image diff stats:  {:s} ".format(arraystats(img_meas_error)))
        print("")
        print("\tAll phases in radians, images total power unity, pupil power constrained to imput value")
        print("\nExample number {:s} ended\n".format(self.tag))
                                                           
        fits.writeto(gs.fileroot+self.tag+"__errorpha.fits", phase_meas_error, overwrite=True)
        fits.writeto(gs.fileroot+self.tag+"__errorint.fits", img_meas_error, overwrite=True)



    def gsloop(self):
        """ The main GS loop.  """
        for iloop in range(self.maxiter + 1):
            if self.iterate() < self.threshold:
                print("\n\tgsloop terminated at iter {}. Threshold sigma is {:.2e} radian \n".format(self.iter, self.threshold))
                break
        self.gswrite()


if __name__ == "__main__":

    # create or obtain pupil with obstructions, apodizations
    #pupilfn = None # creates pupil files, hardcoded single choice of pupill used
    initphase = None # First guess at pupil phase array (radians wavefront) or None
    damp = 1.0  # pupil phase correction  damping, dimensionless, <= 1 # UNUSED for clean didactc demonstration

    # read in or obtain intensity data from in-focus image
    # this data is on the sama angular scale as a lossless dft 
    # of the pupil array, for simplicity.  Trim it appropriately
    # ahead of time if necessary.
    imagefn = None # creates image with a gaussian phase bump aber

    # Save the first 9 iterations, then every 10th iteration
    # eg 1 2 3 ... 9  10 20 30...
    history = (9,10) 

    # rms phase change standard deviation threshold for convergence (radians wavefront)
    threshold = 1.0e-5 # radians

    # max number of iterations (phase, image, back to next phase)
    maxiter = 101


    fileroot = "GSdir/gs"
    dirname = fileroot.split("/")[0]
    print(dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print("Created output directory ", dirname)

    create_input_datafiles(rfn=fileroot) # hardcoded file names for pupil image and true phase files
                                         # These names are used in GS.pupilfn, GS.imagefn, GS.truephasefn
                                         # A convenient output tag is inserted in the output file names of GS.

    for pnum in [0,1,2,3]:

        gs = GerchbergSaxton(pupilfn="{:d}__input_pup.fits".format(pnum),
                             imagefn="{:d}__input_img.fits".format(pnum),
                             truephasefn="{:d}__input_truepha.fits".format(pnum),
                             outputtag="{:d}".format(pnum),  # tags output files with prepended character(s)
                             initphase=None,  # initial guess, rad
                             damp=damp, # pupil phase iteration damping, dimensionless, <= 1 or None (uses 1)
                             history=history, # record of which iterations you want to save for inspection later
                             threshold=threshold, # stop iterating when wavefront rms <= this, rad
                             maxiter=maxiter, # stop iterating after these number of iterations
                             fileroot=fileroot, # file name root for history
                             verbose=False,
                             )
        gs.gsloop()

