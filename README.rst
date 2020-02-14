==============
VesselTrackHFM
==============

Vessel tracking in DCE MR images

Description
===========

A vessel tracking method for DCE-MR liver images based on a data-driven PDE based approach.
This approach is based on the level-sets formulation that produces the (signed) distance function (w.r.t seed-points)
depending on the speed (inverse cost) function used. In our approach we calculate the difference image between the
pre- and post-contrast phases of the DCE-MR image and de-noise it using edge preserving anistropic diffusion filtering.
The speed function is constructed by applying the Frangi vesselness filter on the denoised image. The seed-points
are calculated by thresholding the output of the vesselness filter and and finding the first non-zero components along
all three axes.

The PDE solver based on the Hamiltonian Fast Marching method used can
be found at : https://github.com/Mirebeau/HamiltonFastMarching

The HFM-based PDE solver computes a distance-map w.r.t tubular structures in the image

The vessel tracking method is described in the following papers:

1. 'A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)' by Bekkers et al.
(https://epubs.siam.org/doi/pdf/10.1137/15M1018460)

2. 'Optimal Paths for Variants of the 2D and 3D Reeds-Shepp Car with Applications in Image Analysis' by Duits et al.
(https://link.springer.com/content/pdf/10.1007%2Fs10851-018-0795-z.pdf)


Note
====

This project has been set up using PyScaffold 3.2.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
