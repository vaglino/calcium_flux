import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate

from analyze_experiment_on_surface_v08 import analyze_experiment_on_surface

def setup_directories(base_folder):
    """Set up data and results directories"""
    data_dir = os.path.join(base_folder, 'data')
    res_dir = os.path.join(base_folder, 'results')
    os.makedirs(res_dir, exist_ok=True)
    return data_dir, res_dir


def process_image_stacks(data_dir, res_dir, threshold, gcamp_bleach_params, rfp_bleach_params):
    """Process all image stacks in the data directory"""
    files = [{"folder": data_dir, "name": f} for f in os.listdir(data_dir) if f.endswith('.nd2')]
    
    for image_stack in files:
        print(f"Processing {image_stack['name']}...")
        points, tracks, tracks_raw, tracks_clean, tracks_align = analyze_experiment_on_surface(
            image_stack, threshold,
            gcamp_bleach_params=gcamp_bleach_params,
            rfp_bleach_params=rfp_bleach_params,
            # load_segmentations=True,
            # save_segmentations=False,
            # load_tracking=True,
            # save_tracking=False
            load_segmentations=False,
            save_segmentations=True,
            load_tracking=False,
            save_tracking=True
        )
        
        save_track_results(res_dir, image_stack['name'], tracks, tracks_raw, tracks_clean, tracks_align)

def save_track_results(res_dir, image_name, tracks, tracks_raw, tracks_clean, tracks_align):
    """Save tracking results to CSV files"""
    base_name = os.path.splitext(image_name)[0]
    tracks.to_csv(os.path.join(res_dir, f"{base_name}_tracks.csv"), index=False)
    # tracks_raw.to_csv(os.path.join(res_dir, f"{base_name}_tracks_raw.csv"), index=False)
    # tracks_clean.to_csv(os.path.join(res_dir, f"{base_name}_tracks_clean.csv"), index=False)
    # tracks_align.to_csv(os.path.join(res_dir, f"{base_name}_tracks_align.csv"), index=False)

def analyze_tracks(res_dir):
    """Analyze tracks from all processed files"""
    all_conditions = pd.DataFrame()
    calculated_values = []

    # Create subplots for each metric
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    for file in os.listdir(res_dir):
        if file.endswith("_tracks.csv"):
            tracks = pd.read_csv(os.path.join(res_dir, file))
            condition_data = process_condition(tracks, file)
            all_conditions = pd.concat([all_conditions, condition_data], axis=1)
            
            # Plot GCaMP for this condition
            mean_GCaMP = tracks.groupby('frame_shifted')['MFI_GCaMP_normalized'].mean()
            sem_GCaMP = tracks.groupby('frame_shifted')['MFI_GCaMP_normalized'].sem()
            axs[0].plot(mean_GCaMP.index, mean_GCaMP.values, label=file)
            axs[0].fill_between(mean_GCaMP.index, 
                              (mean_GCaMP - sem_GCaMP).values, 
                              (mean_GCaMP + sem_GCaMP).values, 
                              alpha=0.3)
            
            # Plot RFP for this condition
            mean_RFP = tracks.groupby('frame_shifted')['MFI_RFP_normalized'].mean()
            sem_RFP = tracks.groupby('frame_shifted')['MFI_RFP_normalized'].sem()
            axs[1].plot(mean_RFP.index, mean_RFP.values, label=file)
            axs[1].fill_between(mean_RFP.index, 
                              (mean_RFP - sem_RFP).values, 
                              (mean_RFP + sem_RFP).values, 
                              alpha=0.3)
            
            # Plot ratio for this condition
            mean_ratio = tracks.groupby('frame_shifted')['ratio_normalized'].mean()
            sem_ratio = tracks.groupby('frame_shifted')['ratio_normalized'].sem()
            axs[2].plot(mean_ratio.index, mean_ratio.values, label=file)
            axs[2].fill_between(mean_ratio.index, 
                              (mean_ratio - sem_ratio).values, 
                              (mean_ratio + sem_ratio).values, 
                              alpha=0.3)
            
            calculated_values.append(calculate_statistics(tracks, file))

    axs[0].set_ylabel('Mean Normalized GCaMP')
    axs[0].set_title('Normalized and Aligned GCaMP by Landing Frame')
    axs[0].legend(loc='best')
    
    axs[1].set_ylabel('Mean Normalized RFP')
    axs[1].set_title('Normalized and Aligned RFP by Landing Frame')
    axs[1].legend(loc='best')
    
    axs[2].set_xlabel('Frame Shifted')
    axs[2].set_ylabel('Mean Normalized Ratio (GCaMP/RFP)')
    axs[2].set_title('Normalized and Aligned Ratio by Landing Frame')
    axs[2].legend(loc='best')
    
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(calculated_values)

def process_condition(tracks, file_name):
    """Process a single condition and return its data"""
    # Calculate mean and SEM for all three metrics
    mean_GCaMP = tracks.groupby('frame_shifted')['MFI_GCaMP_normalized'].mean()
    sem_GCaMP = tracks.groupby('frame_shifted')['MFI_GCaMP_normalized'].sem()
    
    mean_RFP = tracks.groupby('frame_shifted')['MFI_RFP_normalized'].mean()
    sem_RFP = tracks.groupby('frame_shifted')['MFI_RFP_normalized'].sem()
    
    mean_ratio = tracks.groupby('frame_shifted')['ratio_normalized'].mean()
    sem_ratio = tracks.groupby('frame_shifted')['ratio_normalized'].sem()
    
    return pd.DataFrame({
        'frame': range(501),
        f'{file_name}_GCaMP_mean': mean_GCaMP.reindex(range(501)).values,
        f'{file_name}_GCaMP_sem': sem_GCaMP.reindex(range(501)).values,
        f'{file_name}_RFP_mean': mean_RFP.reindex(range(501)).values,
        f'{file_name}_RFP_sem': sem_RFP.reindex(range(501)).values,
        f'{file_name}_ratio_mean': mean_ratio.reindex(range(501)).values,
        f'{file_name}_ratio_sem': sem_ratio.reindex(range(501)).values
    }).set_index('frame')

def plot_condition(tracks, file_name):
    """Plot the mean ratio and SEM for a single condition"""
    # Create subplots for each metric
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    # Plot GCaMP
    mean_GCaMP = tracks.groupby('frame_shifted')['MFI_GCaMP_normalized'].mean()
    sem_GCaMP = tracks.groupby('frame_shifted')['MFI_GCaMP_normalized'].sem()
    axs[0].plot(mean_GCaMP.index, mean_GCaMP.values, label=f"{file_name} GCaMP")
    axs[0].fill_between(mean_GCaMP.index, 
                      (mean_GCaMP - sem_GCaMP).values, 
                      (mean_GCaMP + sem_GCaMP).values, 
                      alpha=0.3)
    axs[0].set_ylabel('Normalized GCaMP')
    axs[0].legend()
    
    # Plot RFP
    mean_RFP = tracks.groupby('frame_shifted')['MFI_RFP_normalized'].mean()
    sem_RFP = tracks.groupby('frame_shifted')['MFI_RFP_normalized'].sem()
    axs[1].plot(mean_RFP.index, mean_RFP.values, label=f"{file_name} RFP")
    axs[1].fill_between(mean_RFP.index, 
                      (mean_RFP - sem_RFP).values, 
                      (mean_RFP + sem_RFP).values, 
                      alpha=0.3)
    axs[1].set_ylabel('Normalized RFP')
    axs[1].legend()
    
    # Plot ratio
    mean_ratio = tracks.groupby('frame_shifted')['ratio_normalized'].mean()
    sem_ratio = tracks.groupby('frame_shifted')['ratio_normalized'].sem()
    axs[2].plot(mean_ratio.index, mean_ratio.values, label=f"{file_name} Ratio")
    axs[2].fill_between(mean_ratio.index, 
                      (mean_ratio - sem_ratio).values, 
                      (mean_ratio + sem_ratio).values, 
                      alpha=0.3)
    axs[2].set_ylabel('Normalized Ratio')
    axs[2].set_xlabel('Frame Shifted')
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()

def calculate_statistics(tracks, file_name):
    """Calculate statistics for a single condition"""
    # Calculate statistics for GCaMP
    mean_GCaMP = tracks.groupby('frame_shifted')['MFI_GCaMP_normalized'].mean()
    sem_GCaMP = tracks.groupby('frame_shifted')['MFI_GCaMP_normalized'].sem()
    auc_GCaMP_per_track = tracks.groupby('particle').apply(
        lambda group: integrate.trapz(group['MFI_GCaMP_normalized'].values, group['frame_shifted'].values)
    )
    peak_GCaMP = mean_GCaMP.max()
    time_to_peak_GCaMP = mean_GCaMP.idxmax()
    
    # Calculate statistics for RFP
    mean_RFP = tracks.groupby('frame_shifted')['MFI_RFP_normalized'].mean()
    sem_RFP = tracks.groupby('frame_shifted')['MFI_RFP_normalized'].sem()
    auc_RFP_per_track = tracks.groupby('particle').apply(
        lambda group: integrate.trapz(group['MFI_RFP_normalized'].values, group['frame_shifted'].values)
    )
    peak_RFP = mean_RFP.max()
    time_to_peak_RFP = mean_RFP.idxmax()
    
    # Calculate statistics for ratio
    mean_ratio = tracks.groupby('frame_shifted')['ratio_normalized'].mean()
    sem_ratio = tracks.groupby('frame_shifted')['ratio_normalized'].sem()
    auc_ratio_per_track = tracks.groupby('particle').apply(
        lambda group: integrate.trapz(group['ratio_normalized'].values, group['frame_shifted'].values)
    )
    peak_ratio = mean_ratio.max()
    time_to_peak_ratio = mean_ratio.idxmax()
    
    return {
        'file_name': file_name,
        'auc_GCaMP': auc_GCaMP_per_track.mean(),
        'auc_GCaMP_sem': auc_GCaMP_per_track.sem(),
        'peak_GCaMP': peak_GCaMP,
        'peak_GCaMP_sem': sem_GCaMP[time_to_peak_GCaMP],
        'time_to_peak_GCaMP': time_to_peak_GCaMP,
        'auc_RFP': auc_RFP_per_track.mean(),
        'auc_RFP_sem': auc_RFP_per_track.sem(),
        'peak_RFP': peak_RFP,
        'peak_RFP_sem': sem_RFP[time_to_peak_RFP],
        'time_to_peak_RFP': time_to_peak_RFP,
        'auc_ratio': auc_ratio_per_track.mean(),
        'auc_ratio_sem': auc_ratio_per_track.sem(),
        'peak_ratio': peak_ratio,
        'peak_ratio_sem': sem_ratio[time_to_peak_ratio],
        'time_to_peak_ratio': time_to_peak_ratio
    }

def main():
    # Set up directories
    base_folder = 'Y:/zhu-lab/Stefano/Calcium TGT/BCR TGT GCaMP-TdT/20250309_R_IgM_BSA-NIP/'

    data_dir, res_dir = setup_directories(base_folder)

    # Set analysis parameters
    threshold = 50
    dt = 1  # seconds
    
    # Photobleaching parameters for double exponential model: (A, k1, k2)
    # A is the weight of the first component (between 0 and 1)
    # k1 is the fast decay rate (positive value)
    # k2 is the slow decay rate (positive value)
    
    # Example parameters (these should be replaced with your actual measured values)
    gcamp_bleach_params = (0.106427755, 0.022774652, 5.97145E-05)  # 70% fast component with k=0.005, 30% slow with k=0.001
    rfp_bleach_params = (0.009043184, 0.039693673, 0.000353291)   # 50% fast component with k=0.003, 50% slow with k=0.0005

    # Process image stacks
    process_image_stacks(data_dir, res_dir, threshold, gcamp_bleach_params, rfp_bleach_params)

    # Analyze tracks
    calculated_values_df = analyze_tracks(res_dir)

    # Save results
    calculated_values_df.to_csv(os.path.join(res_dir, 'statistics_values.csv'))
    print("Analysis complete. Results saved in the 'results' directory.")

if __name__ == "__main__":
    main()