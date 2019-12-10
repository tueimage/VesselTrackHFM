from utils.utils import convert_to_grayscale
import SimpleITK as sitk
import numpy as np
import os
from vesseltrackhfm.vesseltrackhfm import VesselTrackHFM

LAMBDA = 500
p = 1.5

IMAGE_DIR = '/home/ishaan/dataset_creation/example/images'

# Read the DCE MR series
dce_pre_contrast = sitk.ReadImage(os.path.join(IMAGE_DIR, 'dce_phase_0.nii'))
dce_post_contrast = sitk.ReadImage(os.path.join(IMAGE_DIR, 'dce_phase_10.nii'))

# Save the metadata
spacing = dce_pre_contrast.GetSpacing()
origin = dce_pre_contrast.GetOrigin()
direction = dce_pre_contrast.GetDirection()

dce_pre_contrast_arr = sitk.GetArrayFromImage(dce_pre_contrast).transpose((1, 2, 0))
dce_post_contrast_arr = sitk.GetArrayFromImage(dce_post_contrast).transpose((1, 2, 0))

# Create subtraction image between post- and pre-contrast phases that highlight the vessels
subtraction_image = np.subtract(dce_post_contrast_arr, dce_pre_contrast_arr)
subtraction_image = np.where(subtraction_image < 0, 0, subtraction_image)

# Track vessels
vessel_tracker = VesselTrackHFM(lmbda=LAMBDA,
                                p=p)

# Compute distance map and geodesic flow
vesselness, distance_map = vessel_tracker(image=subtraction_image)

vesselness_img = sitk.GetImageFromArray(arr=vesselness.transpose((2, 0, 1)))
vesselness_img.SetOrigin(origin)
vesselness_img.SetSpacing(spacing)
vesselness_img.SetDirection(direction)

distance_map_img = sitk.GetImageFromArray(arr=distance_map.transpose((2, 0, 1)))
distance_map_img.SetOrigin(origin)
distance_map_img.SetSpacing(spacing)
distance_map_img.SetDirection(direction)

sitk.WriteImage(vesselness_img, 'vesselness.nii')
sitk.WriteImage(distance_map_img, 'distancemap.nii')

