import SimpleITK as sitk
import numpy as np
import os
from vesseltrackhfm.vesseltrackhfm import VesselTrackHFM
from utils.image_utils import copy_image_metadata
from skimage.filters import threshold_otsu
from dataset_creation.extract_roi import LiverImageCreator

LAMBDA = 500
p = 1.5

IMAGE_DIR = '/home/ishaan/Desktop/UMC_Data/LesionDetection/Detection/Data/1'

# Read the DCE MR series
liver_image_obj = LiverImageCreator(raw_data_dir=IMAGE_DIR,
                                    out_size=256)

dce_img, _, lesion_mask = liver_image_obj.apply_liver_mask()


dce_img_np = sitk.GetArrayFromImage(dce_img).transpose((2, 3, 1, 0))
dce_post_contrast_arr = dce_img_np[:, :, :, 10]
dce_pre_contrast_arr = dce_img_np[:, :, :, 0]

# Create subtraction image between post- and pre-contrast phases that highlight the vessels
subtraction_image = np.subtract(dce_post_contrast_arr, dce_pre_contrast_arr)
subtraction_image = np.where(subtraction_image < 0, 0, subtraction_image)

# Otsu's threhsold on subtraction image
o_thresh = threshold_otsu(image=subtraction_image)

subtraction_image = np.where(subtraction_image > o_thresh, subtraction_image, 0)

# Save the subtraction image
subtraction_img = sitk.GetImageFromArray(arr=subtraction_image.transpose((2, 0, 1)))

subtraction_img.CopyInformation(lesion_mask)

sitk.WriteImage(subtraction_img, 'subtraction_image.nii')

# Track vessels
vessel_tracker = VesselTrackHFM(lmbda=LAMBDA,
                                p=p)

# Compute distance map and geodesic flow
vesselness, distance_map, vesselMask = vessel_tracker(image=subtraction_image)


# Save the outputs
vesselness_img = sitk.GetImageFromArray(arr=vesselness.transpose((2, 0, 1)))

vesselness_img.CopyInformation(lesion_mask)

distance_map_img = sitk.GetImageFromArray(arr=distance_map.transpose((2, 0, 1)))

distance_map_img.CopyInformation(lesion_mask)

vesselMask_img = sitk.GetImageFromArray(arr=vesselMask.transpose((2, 0, 1)))
vesselMask_img.CopyInformation(lesion_mask)

sitk.WriteImage(vesselness_img, 'vesselness.nii')
sitk.WriteImage(distance_map_img, 'distancemap.nii')
sitk.WriteImage(vesselMask_img, 'vesselMask.nii')

