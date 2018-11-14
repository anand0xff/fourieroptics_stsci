## Fourier Optics Classes 1-6: The Basics
#### Anand Sivaramakrishan, STScI, 2018 Oct-Dec

#### Prerequisites: 
	astroconda with python 3 on your laptop
	
	matrixDFT from https://github.com/mperrin/poppy/blob/master/poppy/matrixDFT.py
	(there is a maintained version inside STScI, but we are only using this one routine from poppy.  Get it any way that's convenient.)
	
	Get the python sources utils.py and telclass.py.
	
	Basic familiarity with complex numbers
	
	Familiarity with DS9, image display stretches (linear, log,..) and matplotlib's imshow
	
	Time: 8 hours per week: preparatory work prior to the weekly class, and, post-class, for expanding and writing up work.

#### Course organization

	Pre-class work will get you acquianted with the ideas/questions and try out some exercises.  It is not expected that you will "finish" the problems set for this prep work, but the prep work is essential for the class.  At the start of the class you will all discuss this prep work with each other, and to air your confusion (or clarity) on the concepts and details.  This is a major vehicle of learning in the class.  

	After the in-class discussion I will go over points that need clarification or more in-depth understanding.  That, and setting the stage for the next pre-class prep work will take the remaning part of the class.
	
	After class you should finish the exercises associated with the prep work (e.g. in an annotated Jupyter notebook).  This notebook will serve as your course notes, and hopefully a valuable reference for you and a lasting legacy of the course.
	
	
	

### Class Schedule

| Date        | Class #  | Topics
|:-----------:|:--------:|-------------------------------------------------------------------------|
|Oct 24       | #1      | Intro, class software, EM waves, scalar field, far field, the Fraunhofer approximation behind Fourier optics.  The Fourier transform definition.
|Oct 31       | #2      | The monochromatic plane wave, the single photon probability density function meaning of the PSF.  Limits to discrete Fourier transforms: effects of sampling, and finite information input.  Fourier transform properties (the main theorems), applied to telescopes/imaging (tilts, shifts).  Simulating the image plane of a given telescope pupil.
|Nov 7        | #3      | The Lyot coronagraph.
|Nov 14       | #4      | Aberrations in Lyot coronagraphs. 
|Nov 21       | -       | 
|Nov 28       | #5      | Aberrations in the BLC, Spatially-filtered wavefront sensing
|Dec 5        | #6      | Wavefront sensing: focus sweeps, Gerchberg Saxton focus-diverse phase retrieval basics
|Dec 12       | Back-up | Suggestions for topics of relevance welcome



### Book, articles, and links

1. The best textbook for the Fourier part is still R.N. Bracewell, **"The Fourier Transform and Its Applications" ** (Third Edition, Mcgraw-Hill, 2000).  This book is unfortunately out of print and has been so for some time. The remaining stock is expensive. The ** pictorial dictionary of Fourier transforms** is on pp. 573-591.

2. "Introduction to Fourier Optics"  by J. W. Goodman, (McGraw-Hill, second edition,
    1996 (or any other edition) 
    
3. "Principles of Optics" M. Born & E. Wolf, 1975 (Pergamon). A new edition of this ancient classic was announced a few years ago. Even the older editions have all the basic material you are likely to need. This book is the bible for physical optics.  I use the 7th edition.

4. ["Interferometry and Synthesis in Radio
    Astronomy"](http://adsabs.harvard.edu/abs/2017isra.book.....T) A.R.
    Thompson, J.M. Moran, & G.W. Swenson, (Springer, 2017). This is the bible
    for the subject as applied to radio astronomy. PDF downloadable via the
    link above.

5. [Fourier transforms page at NRAO](https://www.cv.nrao.edu/course/astr534/FourierTransforms.html)  From about 2/3 of the way down and on to the end this fairly short page the basic Fourier transform theorems are laid out well.
   

###Class 1: Theoretical basis, numerical Fourier machinery
	
This is a hands-on course, but you should have some idea of the theoretical basis for numerical manipulations are, in case you need to delve deeper in the future.  We will go quickly over the basics, and outline how we get to the stage of calculating the response of many optical systems to a point source or a sky scene.

We will cover the mathematical definition of the Fourier transform, and a couple of its fundamental properties.

#### Theoretical work	
	
Look up **Fourier transforms** and **Fourier optics**, and [re]familiarize yourself with the key concepts/phrases.

**The 1-dimensional Fourier transform**.  What sort of function (real, complex,. ...) is the function (e.g. f(x)) and its transform (F(k))?  Are the domains of f and F (the spaces that x or k live in) real or complex?  Are f and F, deep down, different descriptions of the same information?
	    
Does the theoretical Fourier transform of a function lose informaton that was present in the original function?    
	
**Incoherent** vs. **coherent** emission of light from distinct points an **extended object**.  Does an atom on one side of a star emit spectral line radiation in phase (coherent with) another atom of the same element on the other side of the star, emitting light in the same spectral line?  Do then two atoms communicate with each other to collaborate on releasing radiation coherent with each  other?  How is this different from a gas in a laser cavity?
				

	
**Diffraction limit** or **[angular] resolution** of a telescope, lambda/D

The Fraunhofer or **far field** situation, and the **Fresnel length** that governs when the far field limit calculations are valid.  Fresnel length ~ D^2 / lambda.  Parse out the meaning of this in terms of the angular resolution.

Describing light numerically/mathematically: a plane wave of monochromatic light (in a homogenous or non-dispersive medium or vacuum) is a propagating oscillation of the electric and magnetic fields.  In (x,y,z) physical space z is the direction of propagation.  (x,y) is the plane transverse to the propagation direction.  The wave can be expressed as the real part of a complex number that has a (real) amplitude A(x,y)  and a unit-strength "phasor" exp( j(kz - wt + phi).  

We average over many periods of the waves' oscillation in a typical optical or IR measurement.  We also "follow a wave's crest", and ignore the (kz - wt) in the exponent.  One resource is [ApJ vol. 552 pp.397-408, 2001 May 1]((https://ui.adsabs.harvard.edu/#abs/2002ApJ...581L..59S)), Section 2.1 equation 1 and surrounding very short text.
	
What mathematical/numerical operation must you perform on a complex array describing the EM wave's "complex amplitude" to get a real array describing the **intensity** (brightness) that a CCD or IR array might detect?
			
The concept of the **pupil** (or aperture) plane of an optical system, and its illumination by **monochromatic light** from a distant star.
	
Describing the pupil quantitatively (eg. equation 2 of the above paper).  If you create a numpy array that represents a circular mirror telescope without a secondary obstruction, what are the physical dimensions you assign (in your mind) to this in-memory array --- what physical thing does the numerical array span?  What physical quantity is represented by a number (an element) in this numerical array?
	
#### Pre-class 2 and class 2 work	

  

1. Become well-acquainted with the **definition of the convolution** of two functions.  How is it applicable to simulating the image of an astronomical scene?
2. Become familiar with the [Special functions](class1/BracewellIII_pp91_92SpecialFunctions.pdf) notations in the repo - we use these in a "cartoon representation" of Fourier theory.
3. Familiarize yourself with the [basic Fourier theorems](https://www.cs.unm.edu/~williams/cs530/theorems6.pdf) which is also in [the repo](class1/theorems6.pdf).
4. Consider what effect a **lateral shift of the pupil** has on its Fourier transform, i.e. the image plane **complex amplitude**, and what effect it has on the **image plane intensity**.  Write down a simple 1-D theoretical Fourier argument to support your conclusions.
5. Again, on purely theoretical grounds, what can you do to the pupil cpmplex amplitude to shift the image intensity array off the origin of the image plane?



Look at and run code/telclass.py
	
	See how the matrixDFT object is instantiated, a pupil created, and an image complex amplitude ("field") and its intensity are calculated.
	
	Look through the parameters of exer1() and its output in datadir/c1/ex1*.fits
	
	Write up a short description of why the images appear as they do.  Consider information content in the two domains, pupil and image.
	
	Do the same with exer2().  How does this show the difference between a detector pixel and a pointwise value of a simulated array that is our image intensity?
	
	After attempting or succeeding in answering questions 4 & 5 above, run exer3(), exer4(), and exer5() by uncommenting the call (if needed) to class2() in the last line of telclass.py.  We will discuss exer3(), exer4(), and exer5() in class.  I put png files of ds9 views of the output fits files in the repo also.
===	
	
### Class 2 plan

- The monochromatic plane wave, the single photon probability density function meaning of the PSF.

- Limits to discrete Fourier transforms: effects of sampling, and finite information input.

-  The basic Fourier theorems.  The Sampling Theorem.

-  Fourier transform properties applied to telescopes/imaging (tilts, shifts, equivalent widths, energy conservation).

- Simulating the image plane of a given telescope pupil.  Pixel scale.  Detector simulation.

- Polychromatic imaging.  


### Class 3 plan

How did I take the photograph in class3/Tilt_in_pupil.jpg?  It is unprocessed.

Understand the details of the Convolution and Similarity theorems in class1/theorms6.pdf.  They will be utilized to understand coronagraphy.  Look at e.g. the effects of convolving one function with another - top hat with a Gaussian curve that is much narrower than the top hat, for example - sketch the result. 

Diffraction-limited stellar coronagraphy [SKMBK2001](https://ui.adsabs.harvard.edu/#abs/2001ApJ...552..397S) especially Section 2.1, Fig. 1 and Fig 2.

Application: Earliest system sketch for GPI in Fig. 9.




### Class 4 prep and in-class work	

Look at these questions individually, think about them and/or attempt them. It could be very helpful for you to get together in groups or subgroups before class 4 and discuss the details amonsgt yourselves.  Some of you have more knowledge of coronagraphs than others, so please share your understanding.

Find the definition of a Band Limited function.  This is used in the perfect [Band-Limited coronagraph](http://iopscience.iop.org/article/10.1086/339625/meta).

Understand how the two graphical derivations in Fig. 2 of the paper [Aberration leak](https://ui.adsabs.harvard.edu/#abs/2005ApJ...634.1416S) work.  One is a classical Lyot coronagraph, the other is a band-limited Lyot coronagraph.  

 - In particular, why is it that to first order, a tilt of the incoming wave (placing the star that needs to be suppressed off-center on the occulting focal plane mask) has no light leak through the coronagraph?
  - Why is it that a quadratic phase error in pupil - a focus phase error - creates what is in effect a faint (and slightly wider) version of the original non-coronagraphic PSF in the final coronagraphic image plane?  Look at Eq. 13, and Fing. 3 in the [Aberration leak](https://ui.adsabs.harvard.edu/#abs/2005ApJ...634.1416S) paper.
   


### Class 5 prep and in-class work	


#### Before class

Low pass and high pass filters in imaging:
 
  - Pupil (stop) filters out **high angular frequency** information in sky image (a.ka. resolution limit).  A finite-sized pupil is a **low pass filter**  A point source is a delta function source in 2D sky angle space (eg RA, Dec).  We only see lower angular frequency components through a finite diameter telescope.

 -  Field stop in an image plane filters out **high spatial frequency** information in pupil. Simulate  Lyot plane intensity with a square hole in the preceding image plane in two cases: send through only three of the Airy rings, or 10 Airy rings).  The field stop is a complement of a Lyot coronagraph's Focal Plane Mask, which lets all high spatial frequency pupil information through but removes the lowest frequencies.
 
Band-limited coronagraph aberrations:

 - Create your own BLC, with a 250x250 pupil array, 100 pixel diameter (not radius) circular pupil inside it, an FPM that is 1 - Jinc^2() where the jinc^2 has its first zero at about the 5th Airy ring of the PSF in the image plane.  Make sure the FPM array iz zero transmission at its center!  FT the FPM*imagefield to the Lyot pupil plane array (another mft.perform() call), and look at the intensity in that Lyot plane **after** you multiply it with a sensible undersized Lyot stop, cutting out the light at the edges of the pupil.
 
 - Also get the final coronagraphic image intensity after one last mft.perform()
 
 - Next, put in a small phase aberratipon into the incoming wavefromt at the entrance pupil.  Put in a tilt, then a parabola centered in the pupil array (i.e. focus), astigmatism, coma, and see what happens in each separate case to the intesity of light in the Lyot pupil.  Make your rms aberration over the active entrance pupil about 0.2 radians (so the first image's Strehl ratio is ~96%), to stay in the small aberration regime.  I put a **Zernike fitting script** that generates a cube of zernikes in the repository - adjust the pupil size in that code to 100 across (match your active pupil), and you can use those slices for your phase.  Check you have your desired phase rms (multiply them by the appropriate constant to ensure that).
 
 Can you reproduce the sort of behavior you see in the  [Aberration leak](https://ui.adsabs.harvard.edu/#abs/2005ApJ...634.1416S) paper?
 
 This is your coronagraphic train - you can make it a Lyot coronagraph by using a 1 - tophat FPM, with the occulting spot about 10 Airy rings in diameter.  You'll have to play with your Lyot stop undersizing to get the Lyot stop effective in this case.
 
 
#### In class

The Sampling theorem (Nyquist/Shannon, original stronger theorem by Laplace) (class6/SamplingTheorem.pdf).

Spatially-filtered wavefront sensing, speckle theory (class6/rjaspeckle.pdf)
	


