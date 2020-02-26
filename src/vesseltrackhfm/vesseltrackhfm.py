"""
A vessel tracking method based on a data-driven PDE based approach.
The PDE solver based on the Hamiltonian Fast Marching method used can
be found at : https://github.com/Mirebeau/HamiltonFastMarching

The HFM-based PDE solver computes a distance-map w.r.t a source point(s) such that distances are small along
tubular structures in the image

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
from skimage.feature.corner import hessian_matrix, _hessian_matrix_image
from skimage.filters import frangi
from scipy.ndimage import generate_binary_structure
from medpy.filter.smoothing import anisotropic_diffusion
import scipy.ndimage.measurements
from scipy.ndimage import label, center_of_mass
from agd import HFMUtils
from agd.Metrics import Riemann
from math import floor
from skimage.filters.ridges import _divide_nonzero, _sortbyabs


class VesselTrackHFM(object):

    def __init__(self,
                 get_distance_map=True,
                 sigmas=(1, 10, 1),
                 alpha=0.5,
                 beta=0.5,
                 gamma=15,
                 lmbda=100,
                 p=1.5,
                 verbose=True,
                 model='isotropic'):
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
        self.model = model

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

        #  Exclude extremal slices while calculating maximum
        max_vesselness_response = np.amax(vessel_filtered_image[:, :, :-10])
        thresh_value = 0.8*max_vesselness_response
        # Use the threshold to create a binary mask
        vesselMask = np.where(vessel_filtered_image >= thresh_value, 1, 0).astype(np.uint8)

        speedR3 = 1 + self.lmbda*np.power(vessel_filtered_image, self.p)
        post_proc_img = np.array(speedR3, dtype=np.float32)
        return post_proc_img, vesselMask

    @staticmethod
    def create_vessel_mask(image):
        #  Exclude extremal slices while calculating maximum
        max_vesselness_response = np.amax(image[:, :, :-10])
        thresh_value = 0.8*max_vesselness_response
        # Use the threshold to create a binary mask
        vesselMask = np.where(image >= thresh_value, 1, 0).astype(np.uint8)
        return vesselMask

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
                se_cc = generate_binary_structure(rank=mask_slice.ndim, connectivity=4)
                labelled_array, num_labels = label(mask_slice, se_cc)
                if num_labels >= 1:
                    seed_slice_idx = slice_idx
                    seed_label_array = labelled_array
                    seed_num_labels = num_labels
                    seed_slice = vesselMask[:, :, seed_slice_idx]

                    # Find largest component among different labels
                    comp_sizes = scipy.ndimage.measurements.sum(input=seed_slice, labels=seed_label_array,
                                                                index=np.arange(1, seed_num_labels + 1))

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

    def _multiscale_hessian_eigenanalysis(self, image=None):
        """
        Analyze eigenvalues and eigenvectors of image hessian at multiple scales
        For each voxel, scale with the best response to the Frangi vesselness filter is chosen
        and eigenvalues and eigenvectors for that scale are returned (along with the vesselness image).
        The eigenvalues returned are sorted (ascending) according to their absolute values

        :param image: (numpy ndarray) 3D image
        :return: hessian_multiscale: (numpy ndarray) Shape: (Y, X, Z, 3, 3) Structure tensor with hessian of
                                     "best scale" for each location
        :return: eigenvals_multiscale: (numpy ndarray) Eigen-values at "best scale" at each location
                                       (sorted by absolute value), shape: (Y, X, Z, 3)
        :return vesselness_multiscale: (numpy ndarray) Frangi vesselness, shape : (Y, X, Z)
        """
        num_steps = floor((self.sigmas[1] - self.sigmas[0]) / self.sigmas[2])

        scale_range = [self.sigmas[0]]
        for step in range(num_steps):
            scale = self.sigmas[0] + min(self.sigmas[1], self.sigmas[0] + step * self.sigmas[2])
            scale_range.append(scale)

        alpha_sq = 2 * self.alpha ** 2
        beta_sq = 2 * self.beta ** 2
        gamma_sq = 2 * self.gamma ** 2
        filtered_array = np.zeros(shape=(len(scale_range), image.shape[0], image.shape[1], image.shape[2]),
                                  dtype=np.float32)

        lambdas_array = np.zeros_like(filtered_array, dtype=np.float32)

        eigenvals_at_all_scales = np.zeros(shape=(len(scale_range), image.shape[0], image.shape[1], image.shape[2], 3),
                                           dtype=np.float32)

        hessian_at_all_scales = np.zeros(shape=(len(scale_range), image.shape[0], image.shape[1], image.shape[2], 3, 3),
                                         dtype=np.float32)

        for idx, sigma in enumerate(scale_range):
            print('Hessian eigen-analysis for scale : {}'.format(sigma))
            H_elems = hessian_matrix(image=image,
                                     sigma=sigma)

            # Correct for scale (Line 148 of ridges.py)
            #
            H_elems = [(sigma ** 2) * e for e in H_elems]

            image_hessian_matrix = _hessian_matrix_image(H_elems)

            eigvalues = np.linalg.eigvalsh(a=image_hessian_matrix)

            eigvalues = _sortbyabs(eigvalues, axis=-1)

            # Transpose to re-use a lot of skimage code (inclduing sorting by value + frangi resp calculation)
            # Frangi code from from skimage: filter/ridges.py line 472
            # We need the frangi response to select the right scale for each voxel position while constructing
            # the metric tensor
            # Swap axes to enable maximum re-use of scipy frangi code
            lambda1, *lambdas = np.transpose(eigvalues, (3, 0, 1, 2))

            r_a = _divide_nonzero(*lambdas) ** 2

            # Compute sensitivity to deviation from a blob-like structure,
            # see equations (10) and (15) in reference [1]_,
            # np.abs(lambda2) in 2D, np.sqrt(np.abs(lambda2 * lambda3)) in 3D
            filtered_raw = np.abs(np.multiply.reduce(lambdas)) ** (1 / len(lambdas))
            r_b = _divide_nonzero(lambda1, filtered_raw) ** 2

            # Compute sensitivity to areas of high variance/texture/structure,
            # see equation (12)in reference [1]_
            r_g = sum([lambda1 ** 2] + [lambdai ** 2 for lambdai in lambdas])

            # Compute output image for given (sigma) scale and store results in
            # (n+1)D matrices, see equations (13) and (15) in reference [1]_
            filtered_array[idx] = ((1 - np.exp(-r_a / alpha_sq)) * np.exp(-r_b / beta_sq) *
                                   (1 - np.exp(-r_g / gamma_sq)))

            eigenvals_at_all_scales[idx] = eigvalues
            hessian_at_all_scales[idx] = image_hessian_matrix
            lambdas_array[idx] = np.max(lambdas, axis=0)  # Store the max value of orthogonal eigenvalues

        filtered_array[lambdas_array > 0] = 0
        arg_max_over_scales = np.argmax(filtered_array, axis=0)

        vesselness_multiscale = np.empty_like(arg_max_over_scales, dtype=np.float32)

        eigenvals_multiscale = np.empty(shape=(image.shape[0], image.shape[1], image.shape[2], 3), dtype=np.float32)

        hessian_multiscale = np.empty(shape=(image.shape[0], image.shape[1], image.shape[2], 3, 3),
                                      dtype=np.float32)

        # TODO: Fix this!!!! Indexing works weird for > 3 dimensions
        # Choose the "best" scale for each voxel location based on response of vesselness filter
        for y in range(arg_max_over_scales.shape[0]):
            for x in range(arg_max_over_scales.shape[1]):
                for z in range(arg_max_over_scales.shape[2]):
                    vesselness_multiscale[y, x, z] = \
                        filtered_array[arg_max_over_scales[y, x, z], y, x, z]

                    eigenvals_multiscale[y, x, z, :] = \
                        eigenvals_at_all_scales[arg_max_over_scales[y, x, z], y, x, z, :]

                    hessian_multiscale[y, x, z, :, :] = \
                        hessian_at_all_scales[arg_max_over_scales[y, x, z], y, x, z, :, :]

        return hessian_multiscale, eigenvals_multiscale, vesselness_multiscale

    @staticmethod
    def calculate_metric_tensor_eigenvalues(eigenvalues):
        """
        Re-map eigenvalues of the image structure tensor (eg: Hessian) to create eigenvalues for the Riemannian
        metric tensor.

        Currently using the mapping shown in Equation 19 of Benmansour and Cohen (2011)
        http://link.springer.com/10.1007/s11263-010-0331-0

        :param eigenvalues: Eigenvalues of image structure tensor
        :return: metric_eigenvalues: Eigenvalues used in the construction of the Riemannian metric tensor
        """
        # lambda0 < lambda1 < lambda2 -- According to Jean-Marie's implementation
        # lambda2 is therefore the eigenvalue along the vessel direction
        eps = 1e-4
        lambda0, lambda1, lambda2 = eigenvalues

        lambda0 = np.add(lambda0, eps)
        lambda1 = np.add(lambda1, eps)
        lambda2 = np.add(lambda2, eps)

        # Calculate \alpha (anisotropy co-efficient) used in Eq (19) in Benmansour and Cohen (2011)
        max_diff = np.amax(lambda2-lambda0)
        anisotropy_coeff = (np.log(5)*4)/max_diff  # mu=5 in paper

        print('Value of anisotropy co-efficient = {}'.format(anisotropy_coeff))

        # Use mapping from Eq. (19) from Benmansour and Cohen (2011)
        mu0 = np.exp(anisotropy_coeff*((lambda1 + lambda2)/2))
        mu1 = np.exp(anisotropy_coeff*((lambda0 + lambda2)/2))
        mu2 = np.exp(anisotropy_coeff*((lambda0 + lambda1)/2))

        return mu0, mu1, mu2

    def create_riemannian_metric_tensor(self, image=None):
        """

        Create metric tensor for a Reimannian manifold such that shortest paths (geodesics) between any 2 points
        lie along vessels

        :param image: (numpy ndarray) 3D image
        :param scales: (tuple) List of sigmas to compute the Hessian over the appropriate range of scales
                              (start, end, step)
        :return: M: (numpy ndarray)
        """

        assert (image.ndim == 3)  # This method works only for 3D images
        assert (len(self.sigmas) == 3)
        hessian_multiscale, eigenvals_multiscale, vesselness = self._multiscale_hessian_eigenanalysis(image=image)

        M_tensor = Riemann.from_mapped_eigenvalues(matrix=hessian_multiscale,
                                                   mapping=self.calculate_metric_tensor_eigenvalues).to_HFM()

        return M_tensor

    @staticmethod
    def sort_eigenvalues_by_mod(eigenValues, eigenVectors):
        """
        Sort eigenvalues by their absolute values and re-organize the eigenvectors in the same
        order

        Modified from: https://stackoverflow.com/questions/43194671/sorted-eigenvalues-and-eigenvectors-on-a-grid
        """

        # Swap the axes for the sorting solution to work
        eigenValues_swapped_axes = eigenValues.transpose((-1, 0, 1, 2))
        eigenVectors_swapped_axes = eigenVectors.transpose((4, 3, 0, 1, 2))

        y, x, z, _ = eigenValues.shape
        sorted_idx = np.argsort(np.abs(eigenValues_swapped_axes), axis=0)  # Last axis
        eigenValues_swapped_axes = np.take_along_axis(eigenValues_swapped_axes, sorted_idx, axis=0)
        eigenVectors_swapped_axes = eigenVectors_swapped_axes[(sorted_idx[:, None, ...], )
                                                              + tuple(np.ogrid[:3, :y, :x, :z])]

        # Swap back the axes to original order for the rest of the program to work
        eigenValues = eigenValues_swapped_axes.transpose((1, 2, 3, 0))
        eigenVectors = eigenVectors_swapped_axes.transpose((2, 3, 4, 1, 0))

        return eigenValues, eigenVectors

    def _solve_pde(self, image=None, vesselMask=None):

        # TODO: Improve and validate seed-point selection
        # <axis>_seed_point -- traveses through slices in a direction perpendicular to the axis to find seed-point

        eff_vesselMask = vesselMask[:, :, :-10]

        z_seed_point = self._find_seed_point(vesselMask=eff_vesselMask,
                                             binarize=False)

        print('Seed-point perpendicular to Z-axis = {}. {}, {}'.format(z_seed_point[1], z_seed_point[0], z_seed_point[2]))
        # Exchange X and Z axes of vessel mask
        x_seed_point = self._find_seed_point(vesselMask=eff_vesselMask.transpose((0, 2, 1)),
                                             binarize=False)
        # Swap it back to the right order
        x_seed_point_swapped = np.array([x_seed_point[0], x_seed_point[2], x_seed_point[1]])

        print('Seed-point perpendicular to X-axis = {}, {}, {}'.format(x_seed_point_swapped[1], x_seed_point_swapped[0],
                                                                       x_seed_point_swapped[2]))

        # Exchange Y and Z axes of vessel mask
        y_seed_point = self._find_seed_point(vesselMask=eff_vesselMask.transpose((2, 1, 0)),
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

        if self.model.lower() == 'isotropic':
            speed_function = 1 + self.lmbda*np.power(image, self.p)
            params = {'model': 'Isotropic3',
                      'arrayOrdering': 'YXZ_RowMajor',
                      'order': 2,
                      'dims': [image.shape[0], image.shape[1], image.shape[2]],
                      # size of a pixel (only for physical dimensions)
                      'gridScale': 1.,
                      'speed': speed_function,
                      'seeds': np.array([z_seed_point, x_seed_point_swapped, y_seed_point_swapped]),
                      'exportValues': self.get_distance_map,
                      'exportGeodesicFlow': 1,
                      'verbosity': verbosity,
                      'showProgress': showProgress}
        elif self.model.lower() == 'riemann':
            metric_tensor = self.create_riemannian_metric_tensor(image=image)

            params = {'model': 'Riemann3',
                      'arrayOrdering': 'YXZ_RowMajor',
                      'order': 2,
                      'dims': [image.shape[0], image.shape[1], image.shape[2]],
                      # size of a pixel (only for physical dimensions)
                      'gridScale': 1.,
                      'metric': metric_tensor,
                      'seeds': np.array([z_seed_point, x_seed_point_swapped, y_seed_point_swapped]),
                      'exportValues': 1,
                      'verbosity': verbosity,
                      'showProgress': showProgress}
        else:
            raise RuntimeError('{} is not a valid model'.format(self.model))

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
        hessian_multiscale, eigenvals_multiscale, vesselness_multiscale = self._multiscale_hessian_eigenanalysis(image)
        vesselMask = self.create_vessel_mask(vesselness_multiscale)

        if self.model.lower() == 'isotropic':
            self._solve_pde(image=vesselness_multiscale, vesselMask=vesselMask)
        elif self.model.lower() == 'riemann':
            self._solve_pde(image=image, vesselMask=vesselMask)
        else:
            raise RuntimeError('{} is not a valid model'.format(self.model))

        if self.get_distance_map > 0:
            distance_map = self.output['values']
        else:
            distance_map = None

        return vesselness_multiscale, distance_map















