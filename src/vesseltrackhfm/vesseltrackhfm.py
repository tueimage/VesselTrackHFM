"""
A vessel tracking method based on a data-driven PDE based approach.
The PDE solver based on the Hamiltonian Fast Marching method used can
be found at : https://github.com/Mirebeau/HamiltonFastMarching

The HFM-based PDE solver computes a distance-map w.r.t tubular structures in the image

The vessel tracking method is described in the following papers:
1. 'A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)' by Bekkers et al.
(https://epubs.siam.org/doi/pdf/10.1137/15M1018460)
2. 'Optimal Paths for Variants of the 2D and 3D Reeds-Shepp Car with Applications in Image Analysis' by Duits et al.
(https://link.springer.com/content/pdf/10.1007%2Fs10851-018-0795-z.pdf)

In this work, we aim to check the feasibility of the method to perform vessel tracking in DCE Liver MR scans.

Author: Ishaan Bhat
Email: ishaan@isi.uu.nl
"""

import numpy as np
from skimage import img_as_float32
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import frangi
from skimage.filters import threshold_otsu
from scipy.ndimage import generate_binary_structure
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage.measurements import sum
from scipy.ndimage import label, center_of_mass
from agd import HFMUtils
from math import floor


class VesselTrackHFM(object):

    def __init__(self,
                 get_distance_map=True,
                 sigmas=(1, 10, 1),
                 alpha=0.5,
                 beta=0.5,
                 gamma=15,
                 lmbda=100,
                 p=1.5,
                 verbose=True):
        """
        Class that defines HFM solver based vessel tracking algorithm

        :param get_distance_map: (bool) Flag, if set to True, the hfm solver computes the vessel distance map
        :param sigmas: (tuple) Range of scales to be used in the Frangi filter (scale_range[0], scale_range[1], step_size)
        :param alpha: (float) Control sensitivity of Frangi filter to diff. between plate-like and line-like structures
        :param beta: (float) Control sensitivity of Frangi filter to deviation from blob like structure
        :param gamma: (float) Control senstivity to background noise
        :param lmbda: (int) Parameter for the speed function used by the solver to sharpen filtered image
        :param p: (float) Parameter for the speed function used by the solver to sharpen filtered image
        :param verbose: (bool) Flag, set true for prints
        """

        if get_distance_map is True:
            self.get_distance_map = 1
        else:
            self.get_distance_map = 0

        self.lmbda = lmbda
        self.p = p
        self.verbose = verbose
        self.sigmas = sigmas
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_optimal_gamma(self, image):
        """
        Compute optimal gamma as 0.5 * max(||hessian_norm(sigma)||) as shown in Frangi et al. (1998)

        :param image: (numpy ndarray)
        :param sigmas: (tuple) Tuple of scales
        :return: gamma_opt: (float)

        """
        hessian_norms = []
        num_steps = floor((self.sigmas[1]-self.sigmas[0])/self.sigmas[2])

        sigmas = [min(self.sigmas[0] + self.sigmas[2]*step, self.sigmas[1]) for step in range(num_steps)]

        for sigma in sigmas:
            h_elements = hessian_matrix(image=image, sigma=sigma)
            h_eigvals = hessian_matrix_eigvals(h_elements)
            # Since the hessian is real and symmetric, frobenius norm is calculated via eigenvalues
            frob_norm = np.sqrt(np.sum(np.array([lmbda**2 for lmbda in h_eigvals])))
            hessian_norms.append(frob_norm)

        gamma_opt = 0.5*max(hessian_norms)

        return gamma_opt

    def enhance_vessels(self, image=None):
        assert (isinstance(image, np.ndarray))
        assert (image.ndim <= 3)

        if image.dtype != np.float32:
            image = img_as_float32(image)

        vessel_filtered_image = frangi(image=image,
                                       sigmas=self.sigmas,
                                       alpha=self.alpha,
                                       beta=self.beta,
                                       gamma=self.gamma,
                                       black_ridges=False)

        print('Max intensity for vesselness output = {}'.format(np.amax(vessel_filtered_image)))

        # Threshold the vesselness image using Otsu's method to calculate the threshold
        thresh = threshold_otsu(image=vessel_filtered_image)
        # Use the threshold to create a binary mask
        vesselMask = np.where(vessel_filtered_image >= thresh, 1, 0).astype(np.uint8)

        speedR3 = 1 + self.lmbda*np.power(vessel_filtered_image, self.p)
        post_proc_img = np.array(speedR3, dtype=np.float32)
        return post_proc_img, vesselMask

    @staticmethod
    def _find_seed_point(vesselMask=None, binarize=False):
        """
        The HFM solver requires seed-points to track vessels, this function automates that process.
        Z-axis (dim=2) in the DCE image is the floor-ceiling axis (w.r.t scanner). Therefore we select a seed-point
        from the top-most (max(z dim)) slice which has a non-zero value in its vessel mask.

        :param vesselMask: (numpy ndarray)
        :param binarize: (bool) If true, the mask pixels are rescaled to have values 0 or 1
        :return: seed_points: (numpy ndarray) Array of seed-point co-ordinates
        """

        # Make the image 1's and 0's
        if binarize is True:
            vesselMask = np.divide(vesselMask, np.amax(vesselMask)).astype(np.uint8)

        _, _, slices = vesselMask.shape

        seed_slice_idx = np.nan
        for slice_idx in np.arange(slices-1, -1, -1):
            # Check if mask contains non-zero locations
            nz_indices = np.nonzero(vesselMask[:, :, slice_idx])
            if nz_indices[0].size != 0 and nz_indices[1].size != 0:
                mask_slice = vesselMask[:, :, slice_idx]
                se_cc = generate_binary_structure(rank=mask_slice.ndim, connectivity=8)
                labelled_array, num_labels = label(mask_slice, se_cc)
                if num_labels > 1:
                    seed_slice_idx = slice_idx
                    seed_label_array = labelled_array
                    seed_num_labels = num_labels
                    seed_slice = vesselMask[:, :, seed_slice_idx]

                    # Find largest component among different labels
                    comp_sizes = sum(input=seed_slice, labels=seed_label_array, index=np.arange(1, seed_num_labels + 1))

                    # Since we ignore label 0 (background), add one to index of largest sum to get "true" label of CC
                    largest_component_label = list(comp_sizes).index(max(comp_sizes)) + 1

                    # This seed-point returns a 2D array with the X and Y co-ordinate of the center-of-mass
                    # of the largest component in the seed_slice
                    slice_seed_point = center_of_mass(input=seed_slice,
                                                      labels=seed_label_array,
                                                      index=largest_component_label)

                    if np.isnan(slice_seed_point[0]) or np.isnan(slice_seed_point[1]):
                        continue
                    else:
                        break
                else:
                    continue

        if np.isnan(seed_slice_idx):
            raise RuntimeError('Unable to find slice with non-zero mask value.')

        # Create the 3D seed-point by appending the slice idx
        seed_point = [slice_seed_point[0], slice_seed_point[1], seed_slice_idx]

        return np.array(seed_point)

    def _solve_pde(self, image=None, vesselMask=None):

        # TODO: Improve and validate seed-point selection
        # <axis>_seed_point -- traveses through slices in a direction perpendicular to the axis to find seed-point
        z_seed_point = self._find_seed_point(vesselMask=vesselMask,
                                             binarize=False)

        print('Seed-point perpendicular to Z-axis = {}. {}, {}'.format(z_seed_point[1], z_seed_point[0], z_seed_point[2]))
        # Exchange X and Z axes of vessel mask
        x_seed_point = self._find_seed_point(vesselMask=vesselMask.transpose((0, 2, 1)),
                                             binarize=False)
        # Swap it back to the right order
        x_seed_point_swapped = np.array([x_seed_point[0], x_seed_point[2], x_seed_point[1]])

        print('Seed-point perpendicular to X-axis = {}, {}, {}'.format(x_seed_point_swapped[1], x_seed_point_swapped[0],
                                                                       x_seed_point_swapped[2]))

        # Exchange Y and Z axes of vessel mask
        y_seed_point = self._find_seed_point(vesselMask=vesselMask.transpose((2, 1, 0)),
                                             binarize=False)
        # Swap it back to the right order
        y_seed_point_swapped = np.array([y_seed_point[2], y_seed_point[1], y_seed_point[0]])

        print('Seed-point perpendicular to the Y-axis = {}, {}, {}'.format(y_seed_point_swapped[1], y_seed_point_swapped[0],
                                                                           y_seed_point_swapped[2]))

        if self.verbose is True:
            verbosity = 2
            showProgress = 1
        else:
            verbosity = 0
            showProgress = 0

        params = {'model': 'Isotropic3',
                  'arrayOrdering': 'YXZ_RowMajor',
                  'order': 2,
                  'dims': [image.shape[0], image.shape[1], image.shape[2]],
                  # size of a pixel (only for physical dimensions)
                  'gridScale': 1.,
                  'speed': image,
                  'seeds': np.array([z_seed_point, x_seed_point_swapped, y_seed_point_swapped]),
                  'exportValues': self.get_distance_map,
                  'exportGeodesicFlow': 1,
                  'verbosity': verbosity,
                  'showProgress': showProgress}

        self.output = HFMUtils.Run(params)

    def __call__(self, image=None):
        """
        Call operator for the VesselTrackHFM class
        Wrapper around pre-processing and the PDE solver

        Given an image, produces a distance map and (optionally) a geodesic flow tensor
        It also returns the post-processed vesselness image (used to compute the speed function for the HFM solver)

        :param image: (numpy ndarray) : Liver MR scan, expected to be 3-D
        :return vesselness_image: (numpy ndarray)
        :return: distance_map: (numpy ndarray) Distance map w.r.t. vessels (in general tubular structures in image)
        :return: geodesic_flows: (numpy ndarray) Geodesic flow vectors [geodesic_flows.ndim =  image.ndim+1]
        """

        # Apply anisotropic diffusion filtering
        # Option 1 corresponds to exponential function of gradient magnitude as conduction co-eff as shown in
        # 'Scale-Space and Edge Detection Using Anisotropic Diffusion' Perona and Malik (1990)
        # Option 3 corresponds to the conduction co-efficient given in 'Robust Anistropic Diffusion' by Black et al.
        # This option seems to fix underflow occurring in certain images
        try:
            image = anisotropic_diffusion(img=image,
                                          niter=10,
                                          kappa=50,
                                          option=3)
        except FloatingPointError:  # Underflow because of large kappa
            print('Underflow ocuured, choosing a smaller kappa')
            image = anisotropic_diffusion(img=image,
                                          niter=10,
                                          kappa=25,
                                          option=3)

        # Pre-processing of the image to highlight vessels/tubular structures
        vesselness_image, vesselMask = self.enhance_vessels(image=image)

        # Solve the eikonal PDE to find shortest geodesics using the HFM method
        self._solve_pde(image=vesselness_image, vesselMask=vesselMask)

        if self.get_distance_map > 0:
            distance_map = self.output['values']
        else:
            distance_map = None

        return vesselness_image, distance_map















