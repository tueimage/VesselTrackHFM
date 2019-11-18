from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
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

# Compute distance map and geodesic flow
vesselness, distance_map, geodesics = vessel_tracker(image=subtraction_image)

h, w, slices = distance_map.shape
# Get the flow-maps for X, Y and Z directions
flowX, flowY, flowZ = geodesics[:, :, :, 0], geodesics[:, :, :, 1], geodesics[:, :, :, 2]

x, y, z = np.meshgrid(np.arange(start=0, stop=h, step=10),
                      np.arange(start=0, stop=w, step=10),
                      np.arange(start=0, stop=slices, step=10))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Left to Right')
ax.set_zlabel('Head to Feet')
ax.set_ylabel('Scanner Z-axis')
ax.set_title('Geodesic Flow diagram')
ax.quiver(x, z, y, flowX[::10, ::10, ::10], flowZ[::10, ::10, ::10], flowY[::10, ::10, ::10],
          normalize=False,
          length=0.3,
          color=['r'])

# Save the distance map
write_niftee_image(image_array=convert_to_grayscale(distance_map, dtype=np.uint16),
                   affine=affine,
                   filename='vessel_distance_map.nii')

write_niftee_image(image_array=convert_to_grayscale(vesselness, dtype=np.uint16),
                   affine=affine,
                   filename='frangi_filtered.nii')

plt.savefig('geodesic_flow.png')
plt.show()




