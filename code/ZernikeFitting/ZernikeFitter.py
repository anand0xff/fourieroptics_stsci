#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
@file py102-example2-zernike.py
@brief Fitting a surface in Python example for Python 102 lecture
@author Tim van Werkhoven (t.i.m.vanwerkhoven@gmail.com)
@url http://python101.vanwerkhoven.org
@date 20111012

Created by Tim van Werkhoven (t.i.m.vanwerkhoven@xs4all.nl) on 2011-10-12
Copyright (c) 2011 Tim van Werkhoven. All rights reserved.

This file is licensed under the Creative Commons Attribution-Share Alike
license versions 3.0 or higher, see
http://creativecommons.org/licenses/by-sa/3.0/

Objectified for easier calling and set-up by anand@stsci.edu  2015
AS: see
http://www.staff.science.uu.nl/~werkh108/docs/teach/2011b_python/course/python102/python_102-print.pdf
and 
http://www.vanwerkhoven.org/teaching.html  esp. pyhon 102

Python 3-ized, print(), //, and xrange->range only anand@stsci.edu  2018
"""


"""
anand@ati.st-guest.org:12  ./ZernikeFitter.py
grid:
[[[-1.000e+00 -1.000e+00 -1.000e+00 -1.000e+00]
  [-5.000e-01 -5.000e-01 -5.000e-01 -5.000e-01]
  [ 0.000e+00  0.000e+00  0.000e+00  0.000e+00]
  [ 5.000e-01  5.000e-01  5.000e-01  5.000e-01]]

 [[-1.000e+00 -5.000e-01  0.000e+00  5.000e-01]
  [-1.000e+00 -5.000e-01  0.000e+00  5.000e-01]
  [-1.000e+00 -5.000e-01  0.000e+00  5.000e-01]
  [-1.000e+00 -5.000e-01  0.000e+00  5.000e-01]]]
grid_rho
[[ 1.414e+00  1.118e+00  1.000e+00  1.118e+00]
 [ 1.118e+00  7.071e-01  5.000e-01  7.071e-01]
 [ 1.000e+00  5.000e-01  0.000e+00  5.000e-01]
 [ 1.118e+00  7.071e-01  5.000e-01  7.071e-01]]
anand@ati.st-guest.org:13  ./ZernikeFitter.py
grid:
[[[-6.667e-01 -6.667e-01 -6.667e-01]
  [ 0.000e+00  0.000e+00  0.000e+00]
  [ 6.667e-01  6.667e-01  6.667e-01]]

 [[-6.667e-01  0.000e+00  6.667e-01]
  [-6.667e-01  0.000e+00  6.667e-01]
  [-6.667e-01  0.000e+00  6.667e-01]]]
grid_rho
[[ 9.428e-01  6.667e-01  9.428e-01]
 [ 6.667e-01  0.000e+00  6.667e-01]
 [ 9.428e-01  6.667e-01  9.428e-01]]

"""


### Libraries

import sys
import astropy.io.fits as pyfits
import numpy as N
from scipy.misc import factorial as fac

### Init functions
def zernike_rad(m, n, rho):
    """
    Calculate the radial component of Zernike polynomial (m, n) 
    given a grid of radial coordinates rho.
    
    >>> zernike_rad(3, 3, 0.333)
    0.036926037000000009
    >>> zernike_rad(1, 3, 0.333)
    -0.55522188900000002
    >>> zernike_rad(3, 5, 0.12345)
    -0.007382104685237683
    """
    
    if (n < 0 or m < 0 or abs(m) > n):
        raise ValueError
    
    if ((n-m) % 2):
        return rho*0.0
    
    pre_fac = lambda k: (-1.0)**k * fac(n-k) / ( fac(k) * fac( (n+m)/2.0 - k ) * fac( (n-m)/2.0 - k ) )
    
    return sum(pre_fac(k) * rho**(n-2.0*k) for k in range((n-m)//2+1))

def zernike(m, n, rho, phi):
    """
    Calculate Zernike polynomial (m, n) given a grid of radial
    coordinates rho and azimuthal coordinates phi.
    
    >>> zernike(3,5, 0.12345, 1.0)
    0.0073082282475042991
    >>> zernike(1, 3, 0.333, 5.0)
    -0.15749545445076085
    """
    if (m > 0): return zernike_rad(m, n, rho) * N.cos(m * phi)
    if (m < 0): return zernike_rad(-m, n, rho) * N.sin(-m * phi)
    return zernike_rad(0, n, rho)

def zernikel(j, rho, phi):
    """
    Calculate Zernike polynomial with Noll coordinate j given a grid of radial
    coordinates rho and azimuthal coordinates phi.
    
    >>> zernikel(0, 0.12345, 0.231)
    1.0
    >>> zernikel(1, 0.12345, 0.231)
    0.028264010304937772
    >>> zernikel(6, 0.12345, 0.231)
    0.0012019069816780774
    """
    n = 0
    while (j > n):
        n += 1
        j -= n
    
    m = -n+2*j
    return zernike(m, n, rho, phi)

def generate_testdata(nzern_, grid_rho_, grid_phi_):
    # hardcoded first 15 zernikes
    test_vec_ = N.random.random(nzern_) - 0.5  # fewer Z's in the test surface...
    test_vec_[15:] = 0.0 # 15 modes excited and rest zero...
    print("input Z coeffts:\n", test_vec_)
    test_surf_ = sum(val * zernikel(i, grid_rho_, grid_phi_) for (i, val) in enumerate(test_vec_))
    return test_vec_, test_surf_


class ZernikeFitter:
    """
    Does Zernikes on a circular disk fitting just inside your array of size narr,
    so if your pupil is undersized within the pupil array, snip off padding before
    sending its wavefront into this object be fit.

    Usage:

        import ZernikeFitter as ZF
        zf = ZF.ZernikeFitter(nzern=10, narr=200)
        zcoeffs, fittedsurface, residualsurface = zf.fit_zernikes_to_surface(yoursurface)


    Zernikes break naturally at nzern = 0 1 3 6 10 15 21 28 36 45 55 66 78 91 105 ...   n*(n+1)/2
    N.B. These are Noll-numbered Zernikes     anand@stsci.edu 2015
    """

    def __init__(self, nzern=15, narr=200, SEEGRID=False, SEEVAR=False, extrasupportindex = None):
        """
        Input: nzern: number of Noll Zernikes to use in the fit
        Input: narr: the live pupil array size you want to use
        
        Sets up list of poly's and grids & support grids
        Makes coordinate grid for rho and phi and circular support mask
        Calculates 'overlap integrals' (covariance matrix) of the Zernike polynomials on your grid and array size
        Calculates the inverse of this matrix, so it's 'ready to fit' your incoming array
        extrasupportindex is a 2D index array to define eg spider supports where the pupil is not live
        
        """
        self.narr = narr
        self.nzern = nzern  # tbd - allowed numbers from Pascal's Triangle sum(n) starting from n=1, viz. n(n+1)/2
        self.grid = (N.indices((self.narr, self.narr), dtype=N.float) - self.narr//2) / (float(self.narr)*0.5) 
        self.grid_rho = (self.grid**2.0).sum(0)**0.5

        if SEEGRID:
            print("grid:")
            print(self.grid)
            print("grid_rho")
            print(self.grid_rho)
            sys.exit()

            
        self.grid_phi = N.arctan2(self.grid[0], self.grid[1])
        self.grid_mask = self.grid_rho <= 1
        self.grid_outside = self.grid_rho > 1

        # Add a spider support sort of extra masking here such as:
        if extrasupportindex:
            # Add a 'spider support':
            self.grid_mask[extrasupportindex] = 0
            self.grid_outside[extrasupportindex] = 1

        # Compute list of explicit Zernike polynomials and keep them around for fitting
        self.zern_list = [zernikel(i, self.grid_rho, self.grid_phi)*self.grid_mask for i in range(self.nzern)]

        
        # Force zernikes to be unit standard deviation over circular mask
        for z, zfunc in enumerate(self.zern_list):
            if z>0: self.zern_list[z] = (zfunc/zfunc[self.grid_mask].std()) * self.grid_mask
            else: self.zern_list[0] = zfunc * self.grid_mask



        #stack = N.zeros((nzern,narr,narr))
        #print "variance:"
        #nolli = 0
        # To normalize all but piston to RMS 1 divide by this number
        self.sigma = []
        for zfunc in self.zern_list:
            self.sigma.append(zfunc[self.grid_mask].std())


        # Calculate covariance between all Zernike polynomials
        self.cov_mat = N.array([[N.sum(zerni * zernj) for zerni in self.zern_list] for zernj in self.zern_list])
        # Invert covariance matrix using SVD
        self.cov_mat_in = N.linalg.pinv(self.cov_mat)


    def fit_zernikes_to_surface(self, surface):
        """
        Input: surface: input surface to be fit (2D array)
        Output: zcoeffs: 1d vector of coefficients of the fit (self.nzern in length)
        Output: rec_wf: the 'recovered wavefront' - i.e. the fitted zernikes, in same array size as surface
        Output: res_wf: surface - rec_wf, i.e. the residual error in the fit

        """

        # Calculate the inner product of each Zernike mode with the test surface
        wf_zern_inprod = N.array([N.sum(surface * zerni) for zerni in self.zern_list])

        # Given the inner product vector of the test wavefront with Zernike basis,
        # calculate the Zernike polynomial coefficients
        zcoeffs = N.dot(self.cov_mat_in, wf_zern_inprod)
        print("First few recovered Zernike coeffts:", zcoeffs[:min(10, self.nzern)])
    
        # Reconstruct (e.g. wavefront) surface from Zernike components
        rec_wf = sum(val * zernikel(i, self.grid_rho, self.grid_phi) for (i, val) in enumerate(zcoeffs))
        rec_wf = rec_wf * self.grid_mask

        print( "Standard deviation of fit is {0:.3e}".format((surface*self.grid_mask - rec_wf)[self.grid_mask].std()) )
        return zcoeffs, rec_wf, (surface*self.grid_mask - rec_wf)




if __name__ == "__main__":

    # This will set up default printing of numpy to 3 dec places, scientific notation
    N.set_printoptions(precision=3, threshold=None, edgeitems=None, linewidth=None,
        suppress=None, nanstr=None, infstr=None, formatter={'float': '{: 0.3e}'.format} )

    #f = ZernikeFitter(nzern=6, narr=4, SEEGRID=True)  
    #zf = ZernikeFitter(nzern=6, narr=3, SEEGRID=True)  
    #sys.exit()

    #### early tests w/exits
    #zf = ZernikeFitter(nzern = 21, narr=201, SEEVAR=True) # Set up to use first 21 Zernikes, radial order 4
    # above exits with sys.exit()

    GEN = True # generate test data 
    GEN = False # read in a kolmog phase screen 

    # initialize ZernikeFitter
    # two test cases are possible - 
    #    test data file or 
    #    test data created in-memory with only the first 15 Zernikes
    #
    if GEN is False: # read in 200 x 200 test data file from disk - adjust path by hand
        test_surf = pyfits.getdata("kol_D_5.00_ro_0.50_DoverR0_10.0l1no0001phase.fits")[:100,:100]
        print(test_surf.shape, "test input file found: {:s}".format("kol_D_5.00_ro_0.50_DoverR0_10.0l1no0001phase.fits"))
        zf = ZernikeFitter(nzern=300, narr=test_surf.shape[0])  # up to radial order 22
    else:
        zf = ZernikeFitter(nzern = 21, narr=201) # Set up to use first 21 Zernikes, radial order 4
        test_vec, test_surf = generate_testdata(zf.nzern, zf.grid_rho, zf.grid_phi) 
        print(test_surf.shape, "test input data generated")

    if True:
        stack = N.zeros((zf.nzern,zf.narr,zf.narr))
        for z, zfunc in enumerate(zf.zern_list):
            if z>0: stack[z,:,:] = (zfunc/zfunc[zf.grid_mask].std())*zf.grid_mask
            else:  stack[z,:,:] = zfunc*zf.grid_mask
        pyfits.PrimaryHDU(data=stack).writeto("ZfunctionsUnitVar.fits", overwrite=True)
    
    # Fit the array with Zernikes:
    zcoeffs, rec_wf, res_wf = zf.fit_zernikes_to_surface(test_surf*zf.grid_mask)

    #test_surf[zf.grid_outside] = 0.0
    #rec_wf[zf.grid_outside] = 0.0
    #res_wf[zf.grid_outside] = 0.0

    # Test reconstruction coeffts for in-memory case (i.e. the fit)
    if GEN:
        print("numpy.allclose(Input, reconstruction coeffs): ",  N.allclose(test_vec, zcoeffs), "\n")


    ### Store data to disk
    from time import time, asctime, gmtime, localtime
    clist = []
    clist.append(pyfits.Card('Program', 'ZernikeFitter.py') )
    clist.append(pyfits.Card('Epoch', time()) )
    clist.append(pyfits.Card('utctime', asctime(gmtime(time()))) )
    clist.append(pyfits.Card('loctime', asctime(localtime(time()))) )
    hdr = pyfits.Header(cards=clist+[pyfits.Card('Desc', 'Surface input')])
    pyfits.writeto('FitZernikes_inputsurface.fits', test_surf, header=hdr, overwrite=True, checksum=True)
    hdr = pyfits.Header(cards=clist+[pyfits.Card('Desc', 'Reconstructed surface')])
    #pyfits.writeto('FitZernikes_fittedsurface.fits', rec_wf*zf.grid_mask, header=hdr, overwrite=True, checksum=True)
    pyfits.writeto('FitZernikes_fittedsurface.fits', rec_wf, header=hdr, overwrite=True, checksum=True)
    if GEN is False:
        hdr = pyfits.Header(cards=clist+[pyfits.Card('Desc', 'Data - Zernfit')])
        pyfits.writeto('FitZernikes_residual.fits', res_wf*zf.grid_mask, header=hdr, overwrite=True, checksum=True)


    ### Plot some results

    import pylab as plt

    #fig = plt.figure(1)
    #ax = fig.add_subplot(111)

    #ax.set_xlabel('Zernike mode')
    #ax.set_ylabel('Power [AU]')
    #ax.set_title('Reconstruction quality')

    #if GEN: ax.plot(test_vec, 'r-', label='Input')
    #ax.plot(zcoeffs, 'b--', label='Recovered')
    #ax.legend()
    #fig.show()
    #fig.savefig('FitZernikes-plot1.pdf')

    #fig = plt.figure(2)
    #fig.clf()
    #ax = fig.add_subplot(111)
    #surf_pl = ax.imshow(rec_wf*zf.grid_mask, interpolation='nearest')
    #fig.colorbar(surf_pl)
    #fig.show()
    #fig.savefig('FitZernikes-plot2.pdf')

    #if GEN is False:
        #fig = plt.figure(3)
        #fig.clf()
        #ax = fig.add_subplot(111)
        #surf_pl = ax.imshow(res_wf*zf.grid_mask, interpolation='nearest')
        #fig.colorbar(surf_pl)
        #fig.show()
        #fig.savefig('FitZernikes-plot3.pdf')
        #print "ds9 FitZernikes_inputsurface.fits using 'invert Y' to compare with FitZernikes-plot2.pdf "

    #print "Hit return to kill all plots: ",
    #raw_input()  # keeps plots up
    #print "\n"
