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

# Feedback from Frank W. : Phase 6 is the post arterial phase with best portal vein visibility
subtraction_image = dce_img_np[:, :, :, 6]

# Save the subtraction image
subtraction_img = sitk.GetImageFromArray(arr=subtraction_image.transpose((2, 0, 1)))
subtraction_img.CopyInformation(lesion_mask)
sitk.WriteImage(subtraction_img, 'subtraction_image_riemann.nii')

vessel_tracker = VesselTrackHFM(lmbda=LAMBDA,
                                p=p,
                                sigmas=(0.3, 2, 0.3),
                                alpha=0.5,
                                beta=0.5,
                                gamma=15,
                                model='riemann'
                                )

vesselness, distance_map, _ = vessel_tracker(image=subtraction_image)


# Save the outputs using the un-modified code
vesselness_img = sitk.GetImageFromArray(arr=vesselness.transpose((2, 0, 1)))
vesselness_img.CopyInformation(lesion_mask)

distance_map_img = sitk.GetImageFromArray(arr=distance_map.transpose((2, 0, 1)))
distance_map_img.CopyInformation(lesion_mask)

sitk.WriteImage(vesselness_img, 'vesselness_riemann.nii')
sitk.WriteImage(distance_map_img, 'distancemap_riemann.nii')


