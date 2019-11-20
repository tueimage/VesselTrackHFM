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
from skimage.filters import gaussian
from skimage.filters import frangi
from skimage.filters import threshold_otsu
from scipy.ndimage import generate_binary_structure, binary_erosion, binary_dilation
from scipy.ndimage.measurements import sum
from scipy.ndimage import label, center_of_mass
import sys
# FIXME: Conda install for the 'agd' package(https://github.com/Mirebeau/AdaptiveGridDiscretizations) is broken.
# Appending path to cloned repo for run-time inclusion of the package
sys.path.append('/home/ishaan/agd/AdaptiveGridDiscretizations')
from agd import HFMUtils


class VesselTrackHFM(object):

    def __init__(self, get_distance_map=True, lmbda=100, p=1.5,
                 verbose=True):
        """
        Class that defines HFM solver based vessel tracking algorithm

        :param hfm_solver: (HFMIO object) HFM solver (eg: Isotropic3, Riemannian, Reeds-Shepp etc.)
        :param out_dir:
        :param get_distance_map: (bool) Flag, if set to True, the hfm solver computes the vessel distance map
        :param get_geodesic_flow: (bool) Flag, if set to True, the hfm solver produces geodesic flow maps
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

    @staticmethod
    def _preprocess_image(image=None):
        assert (isinstance(image, np.ndarray))
        assert (image.ndim <= 3)

        if image.dtype != np.float32:
            image = img_as_float32(image)

        # Smooth image to remove high freq noise
        # TODO: Better de-noising method specially tailored for DCE MR images
        smoothed_image = gaussian(image=image,
                                  sigma=1.5)

        #  Enhance vessels by using Frangi filter
        vessel_filtered_image = frangi(image=smoothed_image,
                                       black_ridges=False)

        # Threshold the vesselness image using Otsu's method to calculate the threshold
        thresh = threshold_otsu(image=vessel_filtered_image)
        # Use the threshold to create a binary mask
        vesselMask = np.where(vessel_filtered_image >= thresh, 1, 0).astype(np.uint8)

        # Disabling post-processing of the Frangi filtered image (on Erik's suggestion)
        # # Morphological operations on the binarized vessel mask
        # se_erosion = generate_binary_structure(rank=vesselMask.ndim, connectivity=2)
        # se_dilation = generate_binary_structure(rank=vesselMask.ndim, connectivity=8)
        #
        # #  Perform erosion-dilation instead of opening to have better fine grain control
        # #  (iterations, different structure elements etc.)
        # vesselMask_erosion = binary_erosion(input=vesselMask, structure=se_erosion, iterations=2).astype(vesselMask.dtype)
        # vesselMask_opened = binary_dilation(input=vesselMask_erosion, structure=se_dilation, iterations=3).astype(vesselMask.dtype)
        #
        # # Apply post-processed mask to the vesselness image to get rid of the noise
        # post_proc_img = np.multiply(vessel_filtered_image, vesselMask_opened)

        post_proc_img = vessel_filtered_image
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
                seed_slice_idx = slice_idx
                break

        if np.isnan(seed_slice_idx):
            raise RuntimeError('Unable to find slice with non-zero mask value.')

        seed_slice = vesselMask[:, :, seed_slice_idx]

        se_cc = generate_binary_structure(rank=seed_slice.ndim, connectivity=8)

        labelled_array, num_labels = label(seed_slice, se_cc)

        # Find largest component among different labels
        comp_sizes = sum(input=seed_slice, labels=labelled_array, index=np.arange(1, num_labels+1))
        largest_component_label = list(comp_sizes).index(max(comp_sizes))

        # This seed-point returns a 2D array with the X and Y co-ordinate of the center-of-mass
        # of the largest component in the seed_slice
        slice_seed_point = center_of_mass(input=seed_slice,
                                          labels=labelled_array,
                                          index=largest_component_label)

        # Create the 3D seed-point by appending the slice idx
        seed_point = [slice_seed_point[0], slice_seed_point[1], seed_slice_idx]

        return np.array(seed_point)

    def _solve_pde(self, image=None, vesselMask=None):

        seed_points = self._find_seed_point(vesselMask=vesselMask,
                                            binarize=False)

        # Construct the speed function for the PDE
        speedR3 = np.divide(image, np.amax(image)).astype(np.float32)
        speedR3 = 1 + self.lmbda*np.power(speedR3, self.p)

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
                  'speed': speedR3,
                  'seeds': np.array([seed_points]),
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
        # Pre-processing of the image to highlight vessels/tubular structures
        vesselness_image, vesselMask = self._preprocess_image(image=image)

        # Solve the eikonal PDE to find shortest geodesics using the HFM method
        self._solve_pde(image=vesselness_image, vesselMask=vesselMask)

        if self.get_distance_map > 0:
            distance_map = self.output['values']
        else:
            distance_map = None

        return vesselness_image, distance_map















