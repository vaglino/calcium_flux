import numpy as np
import pandas as pd
import os
import nd2
import matplotlib.pyplot as plt
from skimage.io import imread
from tifffile import imwrite, TiffFile, TiffWriter

from scipy.ndimage import uniform_filter1d
from segmentation import segment_cells
from cell_parameters import extract_cell_parameters
import time




def segment_all_frames(stack):
    img_file = os.path.join(stack['folder'], stack['name'])
    
    results_dir = os.path.join(stack['folder'], 'results')
    os.makedirs(results_dir, exist_ok=True)

    segmented_images_path = os.path.join(results_dir, f"{stack['name'].replace('.nd2', '')}_segmentation.tif")

    with nd2.ND2File(img_file) as nd2_file:
        print(f"Number of frames: {nd2_file.sizes['T']}")
        print(f"Number of channels: {nd2_file.sizes['C']}")
        print(f"Image shape: {nd2_file.sizes['Y']}x{nd2_file.sizes['X']}")

        n_frames = nd2_file.sizes['T']
        saved_params = [None] * n_frames
        points = [None] * n_frames
        # saved_params = [None] * 100
        # points = [None] * 100
        
        with TiffWriter(segmented_images_path, bigtiff=True) as tif_writer:
            for i in range(n_frames):
            # for i in range(100):


                if i % 1 == 0:
                    print(f'Segmenting frame {i}/{n_frames}')

                # Access RFP and GCaMP channels
                frame = nd2_file.read_frame(i)
                I_GCaMP = frame[0]  # Assuming GCaMP is the first channel
                I_RFP = frame[1]  # Assuming RFP is the second channel


                # show image I_RFP
                # plt.imshow(I_RFP)
                # plt.show()
                # # show image I_GCaMP
                # plt.imshow(I_GCaMP)
                # plt.show()
                #crop images to 512x512
                # I_RFP = I_RFP[300:812, 300:812]
                # I_GCaMP = I_GCaMP[300:812, 300:812]


                # Segment cells based on RFP channel
                I_seg = segment_cells(I_RFP, plotting=False)
                
                # Extract parameters using both channels
                params = extract_cell_parameters(I_RFP, I_GCaMP, I_seg)

                centroids = params[:, 1:3]
                # Debugging statements to check the shape and type of params and saved_params
                # print(f"Index: {i}")
                # print(f"Type of params: {type(params)}")
                # print(f"Shape of params: {np.shape(params) if isinstance(params, np.ndarray) else 'N/A'}")
                # print(f"Type of saved_params: {type(saved_params)}")
                # print(f"Shape of saved_params: {np.shape(saved_params) if isinstance(saved_params, np.ndarray) else 'N/A'}")

                saved_params[i] = params
                points[i] = centroids
                tif_writer.save(I_seg)#, compress=0)

    return saved_params, points

