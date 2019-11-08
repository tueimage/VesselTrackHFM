from utils.image_utils import write_niftee_image, read_niftee_image
from utils.utils import convert_to_grayscale
import numpy as np
from HFMpy import HFM_Isotropic3
from vesseltrackhfm.vesseltrackhfm import VesselTrackHFM

IMAGE_PATH = '/home/ishaan/umc_data/lesion_dataset_clean/train/extracted_roi/47/centered_dce_liver_47_0.nii'

# Use the Isotropic3 solver since we consider the image in the R3 space
hfm_solver = HFM_Isotropic3.HFMIO()

# Read the DCE MR series
dce_mr_series, affine = read_niftee_image(filename=IMAGE_PATH)

# Create subtraction image between post- and pre-contrast phases that highlight the vessels
subtraction_image = np.subtract(dce_mr_series[:, :, :, 4], dce_mr_series[:, :, :, 0])
subtraction_image = np.where(subtraction_image < 0, 0, subtraction_image)

# Save the subtraction image
write_niftee_image(image_array=convert_to_grayscale(subtraction_image, dtype=np.uint16),
                   affine=affine,
                   filename='subtraction_image.nii')

# Track vessels
vessel_tracker = VesselTrackHFM(hfm_solver=hfm_solver,
                                lmbda=100,
                                p=1.5)

pre_proc_img, distance_map, geodesic_flow = vessel_tracker(image=subtraction_image)

write_niftee_image(image_array=convert_to_grayscale(pre_proc_img, dtype=np.uint16),
                   affine=affine,
                   filename='post_proc_vessel_img.nii')

# Save the distance map
write_niftee_image(image_array=convert_to_grayscale(distance_map, dtype=np.uint16),
                   affine=affine,
                   filename='vessel_distance_map.nii')





