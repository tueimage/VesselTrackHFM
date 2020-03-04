import SimpleITK as sitk
import numpy as np
from vesseltrackhfm.vesseltrackhfm import VesselTrackHFM
from dataset_creation.extract_roi import LiverImageCreator

LAMBDA = 1e4
p = 1.0

IMAGE_DIR = '/home/ishaan/Desktop/UMC_Data/Data/29/20150518'

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

# Save the subtraction image
subtraction_img = sitk.GetImageFromArray(arr=subtraction_image.transpose((2, 0, 1)))
subtraction_img.CopyInformation(lesion_mask)
sitk.WriteImage(subtraction_img, 'subtraction_image_isotropic.nii')

# Liver veins (portal and hepatic) have diameters in the range 2-26mm
# In voxel size, this roughly corresponds to a range of 1-17 voxels
# For a given sigma, only voxels in a 3*sigma radius contribute meaningfully to the convolution
# Given the range of diameters and image spacing, sigma should be chosen in the range -- [1, 3], technically
# sigma should start with a value less than 1, but that also introduces a lot of small structures/false positives
# in the filter output
vessel_tracker = VesselTrackHFM(lmbda=LAMBDA,
                                p=p,
                                sigmas=(0.3, 3, 0.3),
                                alpha=0.5,
                                beta=0.5,
                                gamma=15,
                                model='isotropic'
                                )

# Compute distance map and geodesic flow
vesselness, distance_map, _ = vessel_tracker(image=subtraction_image)


# Save the outputs
vesselness_img = sitk.GetImageFromArray(arr=vesselness.transpose((2, 0, 1)))
vesselness_img.CopyInformation(lesion_mask)

distance_map_img = sitk.GetImageFromArray(arr=distance_map.transpose((2, 0, 1)))
distance_map_img.CopyInformation(lesion_mask)

sitk.WriteImage(vesselness_img, 'vesselness_isotropic.nii')
sitk.WriteImage(distance_map_img, 'distancemap_isotropic.nii')
