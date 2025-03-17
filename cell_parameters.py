
import numpy as np
from scipy.ndimage import measurements

def extract_cell_parameters(I_RFP, I_GCaMP, I_seg):
    """
    Extract various parameters for each cell in the segmented image.

    Args:
    I_RFP (ndarray): RFP channel image
    I_GCaMP (ndarray): GCaMP channel image
    I_seg (ndarray): Segmented image with labeled cells

    Returns:
    ndarray: Array of cell parameters with columns:
        [cell_id, centroid_y, centroid_x, MFI_RFP, MFI_GCaMP, ratio, area]
    """
    labels, n_cells = measurements.label(I_seg)

    if n_cells == 0:
        return np.empty((0, 7))  # Return empty array with 7 columns

    centroid = measurements.center_of_mass(I_RFP, labels, np.arange(1, n_cells + 1))
    MFI_RFP = measurements.mean(I_RFP, labels, np.arange(1, n_cells + 1))
    MFI_GCaMP = measurements.mean(I_GCaMP, labels, np.arange(1, n_cells + 1))
    ratio = MFI_GCaMP / MFI_RFP
    area = measurements.sum(I_seg, labels, np.arange(1, n_cells + 1))

    params = np.column_stack((
        np.arange(1, n_cells + 1),
        np.array(centroid),
        MFI_RFP,
        MFI_GCaMP,
        ratio,
        area
    ))
    
    return params
# import numpy as np
# from scipy.ndimage import measurements


# def extract_cell_parameters(I_RFP, I_GCaMP, I_seg):
#     labels, n_cells = measurements.label(I_seg)

#     if n_cells == 0:
#         return np.empty((0, 6))  # Changed from 5 to 6 to include ratio

#     centroid = measurements.center_of_mass(I_RFP, labels, np.arange(1, n_cells + 1))
    
#     MFI_RFP = measurements.mean(I_RFP, labels, np.arange(1, n_cells + 1))
#     MFI_GCaMP = measurements.mean(I_GCaMP, labels, np.arange(1, n_cells + 1))
    
#     # Calculate ratio of GCaMP to RFP
#     ratio = MFI_GCaMP / MFI_RFP

#     area = measurements.sum(I_seg, labels, np.arange(1, n_cells + 1))

#     params = np.hstack((np.arange(1, n_cells + 1)[:, None], np.array(centroid), MFI_RFP[:, None], MFI_GCaMP[:, None], ratio[:, None], area[:, None]))
    
#     return params



# def extract_cell_parameters(I, I_seg):
#     labels, n_cells = measurements.label(I_seg)

#     # If there are no cells, return an empty array or some default value
#     if n_cells == 0:
#         return np.empty((0, 5))

#     # Find the centroid using scipy's center_of_mass function
#     centroid = measurements.center_of_mass(I, labels, np.arange(1, n_cells + 1))
    
#     # Calculate MFI (Mean Fluorescence Intensity)
#     MFI = measurements.mean(I, labels, np.arange(1, n_cells + 1))

#     # Calculate the area of each cell
#     area = measurements.sum(I_seg, labels, np.arange(1, n_cells + 1))

#     # Save results
#     params = np.hstack((np.arange(1, n_cells + 1)[:, None], np.array(centroid), MFI[:, None], area[:, None]))
    
#     return params