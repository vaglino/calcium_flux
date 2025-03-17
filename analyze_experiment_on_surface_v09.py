import numpy as np
import os
import pandas as pd
import pickle
import trackpy as tp
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, savgol_filter
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec

from stack2tracks_nd2 import segment_all_frames
from visualization import visualize_segmentation

def analyze_experiment_on_surface(stack, threshold, 
                                gcamp_bleach_params=(0.5, 0, 0), 
                                rfp_bleach_params=(0.5, 0, 0),
                                save_segmentations=True, load_segmentations=False, 
                                save_tracking=True, load_tracking=False):
    """
    Analyze an experiment on surface by segmenting cells, tracking them, and processing their fluorescence data.

    Args:
    stack (dict): Dictionary containing 'folder' and 'name' of the image stack
    threshold (float): Threshold value for cell detection
    gcamp_bleach_params (tuple): (A, k1, k2) parameters for GCaMP double exponential photobleaching
    rfp_bleach_params (tuple): (A, k1, k2) parameters for RFP double exponential photobleaching
    save_segmentations (bool): Whether to save segmentation results
    load_segmentations (bool): Whether to load 
    
    pre-computed segmentations
    save_tracking (bool): Whether to save tracking results
    load_tracking (bool): Whether to load pre-computed tracking results

    Returns:
    tuple: (points, tracks, tracks_raw, tracks_clean, tracks_align)
    """
    base_name = os.path.splitext(stack['name'])[0]
    saved_params_filename = os.path.join(stack['folder'], f'{base_name}_saved_params.npy')
    points_filename = os.path.join(stack['folder'], f'{base_name}_points.npy')
    tracks_filename = os.path.join(stack['folder'], f'{base_name}_tracking.csv')

    # Load or compute segmentations
    saved_params, points = load_or_compute_segmentations(stack, load_segmentations, save_segmentations,
                                                         saved_params_filename, points_filename)
    
    # Load or compute tracking
    tracks = load_or_compute_tracking(load_tracking, save_tracking, tracks_filename, points, saved_params)

    # Process tracks
    tracks_raw = tracks.copy()
    tracks = process_tracks(tracks)
    

    tracks_clean = tracks.copy()
    tracks_align = tracks.copy()

    # Smooth and normalize tracks
    # Pass bleaching parameters to smooth_and_normalize_tracks
    tracks = smooth_and_normalize_tracks(tracks, gcamp_bleach_params, rfp_bleach_params)

    # Plot results
    print('Plotting normalized MFI...')
    plot_normalized_mfi(tracks)

    print('Analysis complete.')

    return points, tracks, tracks_raw, tracks_clean, tracks_align

def load_or_compute_segmentations(stack, load_segmentations, save_segmentations, saved_params_filename, points_filename):
    """Load pre-computed segmentations or compute new ones"""
    if load_segmentations:
        print('Loading precomputed segmentations...')
        saved_params = np.load(saved_params_filename, allow_pickle=True)
        with open(points_filename, 'rb') as f:
            points = pickle.load(f)
    else:
        print('Performing segmentation...')
        saved_params, points = segment_all_frames(stack)
        saved_params = np.array(saved_params, dtype=object)
        if save_segmentations:
            print('Saving segmentations...')
            np.save(saved_params_filename, saved_params, allow_pickle=True)
            with open(points_filename, 'wb') as f:
                pickle.dump(points, f)
    return saved_params, points


def load_or_compute_tracking(load_tracking, save_tracking, tracks_filename, points, saved_params):
    """Load pre-computed tracking results or compute new ones"""
    if load_tracking:
        print('Loading precomputed tracking...')
        tracks = pd.read_csv(tracks_filename)
    else:
        print('Tracking cells...')
        tracks = compute_tracking(points, saved_params)
        if save_tracking:
            print('Saving tracking...')
            tracks.to_csv(tracks_filename, index=False)
    return tracks

def compute_tracking(points, saved_params):
    """Compute cell tracking"""
    frames = []
    for frame_num, (pts, params) in enumerate(zip(points, saved_params)):
        df = pd.DataFrame(pts, columns=['y', 'x'])  
        df[['p1', 'p2', 'p3', 'MFI_RFP', 'MFI_GCaMP', 'ratio', 'area']] = pd.DataFrame(params)
        df['frame'] = frame_num
        df['particle'] = df.index
        frames.append(df)
    points_df = pd.concat(frames)

    max_linking_distance = 30
    max_gap_closing = 20
    return tp.link_df(points_df, max_linking_distance, memory=max_gap_closing)

def process_tracks(tracks):
    """Process tracks by filtering and aligning"""
    tracks = tp.filter_stubs(tracks, 600)
    print('Before:', tracks['particle'].nunique())
    tracks = tracks.groupby('particle').filter(lambda x: x['frame'].min() > 0)
    print('After landing:', tracks['particle'].nunique())
    # This creates a temporary frame_shifted column based on first appearance
    # This will be replaced later after landing frame detection
    tracks['frame_shifted'] = tracks.groupby('particle')['frame'].transform(lambda x: x - x.min())
    # print(tracks.columns)
    return tracks


def smooth_track(track):
    """Apply uniform filter to smooth the track"""
    track_smooth = uniform_filter1d(track, size=10)
    return uniform_filter1d(track_smooth, size=10)


def correct_photobleaching(tracks, gcamp_params, rfp_params):
    """
    Correct photobleaching in GCaMP and RFP signals using double exponential decay model.
    Uses the landing frame as the reference point for photobleaching correction.
    
    Args:
    tracks (pd.DataFrame): DataFrame containing track data with frame_shifted column
                          where frame_shifted=0 corresponds to the landing frame
    gcamp_params (tuple): (A, k1, k2) parameters for GCaMP photobleaching
    rfp_params (tuple): (A, k1, k2) parameters for RFP photobleaching
    
    Returns:
    pd.DataFrame: DataFrame with corrected GCaMP, RFP, and ratio values
    """
    # Make a copy of the tracks DataFrame
    tracks_corrected = tracks.copy()
    
    # Unpack parameters
    gcamp_A, gcamp_k1, gcamp_k2 = gcamp_params
    rfp_A, rfp_k1, rfp_k2 = rfp_params
    
    # Only process frames at or after landing
    # Note: frame_shifted now represents frames relative to landing frame
    post_landing_mask = tracks_corrected['frame_shifted'] >= 0
    
    if post_landing_mask.any():
        # Get unique frame_shifted values (at or after landing)
        unique_frames = tracks_corrected.loc[post_landing_mask, 'frame_shifted'].unique()
        
        # Pre-compute correction factors for all possible frame_shifted values
        frame_correction_map = {}
        for frame_shifted in unique_frames:
            # Calculate decay factors
            gcamp_decay = gcamp_A * np.exp(-gcamp_k1 * frame_shifted) + (1 - gcamp_A) * np.exp(-gcamp_k2 * frame_shifted)
            rfp_decay = rfp_A * np.exp(-rfp_k1 * frame_shifted) + (1 - rfp_A) * np.exp(-rfp_k2 * frame_shifted)
            
            # Calculate correction factors (inverse of decay)
            gcamp_correction = 1.0 / gcamp_decay if gcamp_decay > 0 else 1.0
            rfp_correction = 1.0 / rfp_decay if rfp_decay > 0 else 1.0
            
            frame_correction_map[frame_shifted] = (gcamp_correction, rfp_correction)
        
        # Create correction factor arrays using map lookup
        tracks_corrected.loc[post_landing_mask, 'gcamp_correction'] = \
            tracks_corrected.loc[post_landing_mask, 'frame_shifted'].map(
                lambda x: frame_correction_map[x][0])
        
        tracks_corrected.loc[post_landing_mask, 'rfp_correction'] = \
            tracks_corrected.loc[post_landing_mask, 'frame_shifted'].map(
                lambda x: frame_correction_map[x][1])
        
        # Apply corrections in one vectorized operation
        tracks_corrected.loc[post_landing_mask, 'MFI_GCaMP'] = \
            tracks_corrected.loc[post_landing_mask, 'MFI_GCaMP'] * \
            tracks_corrected.loc[post_landing_mask, 'gcamp_correction']
            
        tracks_corrected.loc[post_landing_mask, 'MFI_RFP'] = \
            tracks_corrected.loc[post_landing_mask, 'MFI_RFP'] * \
            tracks_corrected.loc[post_landing_mask, 'rfp_correction']
        
        # Drop temporary columns
        tracks_corrected.drop(['gcamp_correction', 'rfp_correction'], axis=1, inplace=True)
    
    # Recalculate ratio after correction
    tracks_corrected['ratio'] = tracks_corrected['MFI_GCaMP'] / tracks_corrected['MFI_RFP']
    
    return tracks_corrected

def calculate_first_derivative(x):
    """Calculate the first derivative of the smoothed MFI"""
    diff = np.diff(x)
    return np.pad(diff, (0, 1), 'constant', constant_values=0)


def detect_landing_frame(track, max_frames=100, plateau_threshold=0.05, window_size=7, polyorder=2):
    """
    Detect the landing frame based on when RFP signal first reaches a plateau.
    
    Args:
    track (pd.DataFrame): DataFrame containing track data for a single particle
    max_frames (int): Maximum number of frames to consider for landing detection
    plateau_threshold (float): Threshold for considering the derivative small enough for a plateau
    window_size (int): Window size for Savitzky-Golay filter
    polyorder (int): Polynomial order for Savitzky-Golay filter
    
    Returns:
    int: Detected landing frame
    """
    # Limit search to the first max_frames
    track_subset = track.iloc[:min(max_frames, len(track))]
    
    if len(track_subset) < 10:  # Need enough points for reliable detection
        return track_subset.index[0]  # Return first frame as fallback
    
    # Get the RFP signal
    rfp_signal = track_subset['MFI_RFP'].values
    frames = track_subset['frame'].values
    
    # Smooth the RFP signal
    try:
        rfp_smooth = savgol_filter(rfp_signal, window_size, polyorder)
    except ValueError:  # If window_size is too large for the data
        # Fall back to smaller window size
        new_window_size = min(5, len(rfp_signal) - 2)
        if new_window_size % 2 == 0:  # Must be odd
            new_window_size = max(3, new_window_size - 1)
        rfp_smooth = savgol_filter(rfp_signal, new_window_size, min(1, polyorder))
    
    # Calculate the derivative of the smoothed signal
    rfp_derivative = np.gradient(rfp_smooth)
    
    # Normalize the derivative for easier thresholding
    if np.max(rfp_smooth) - np.min(rfp_smooth) > 0:
        normalized_derivative = rfp_derivative / (np.max(rfp_smooth) - np.min(rfp_smooth))
    else:
        return track_subset.index[0]  # No significant change, return first frame
    
    # Find where the derivative drops below the threshold
    # We're looking for where the signal stops increasing rapidly
    plateau_candidates = np.where(normalized_derivative < plateau_threshold)[0]
    
    if len(plateau_candidates) == 0:
        return track_subset.index[0]  # No plateau found, return first frame
    
    # Find the first point after initial rapid rise where derivative is small
    # First, find where the signal has risen significantly from baseline
    baseline = np.mean(rfp_smooth[:min(5, len(rfp_smooth))])  # Mean of first few frames
    plateau_level = np.mean(rfp_smooth[-min(10, len(rfp_smooth)):])  # Mean of last few frames
    
    # Find frames where signal is above 80% of the way from baseline to plateau
    # This helps avoid detecting noise in the baseline as a plateau
    rise_threshold = baseline + 0.8 * (plateau_level - baseline)
    rise_frames = np.where(rfp_smooth >= rise_threshold)[0]
    
    if len(rise_frames) == 0:
        return track_subset.index[0]  # No significant rise, return first frame
    
    # Find the first plateau candidate after significant rise
    valid_plateaus = [p for p in plateau_candidates if p >= rise_frames[0]]
    
    if len(valid_plateaus) == 0:
        return track_subset.index[0]  # No valid plateau, return first frame
    
    # Return the corresponding frame
    landing_frame_idx = valid_plateaus[0]
    if landing_frame_idx < len(frames):
        return track_subset.index[landing_frame_idx]
    else:
        return track_subset.index[0]  # Fallback to first frame
    
def normalize_by_landing(track):
    """
    Normalize MFI by the value at or closest to the landing frame.
    
    Args:
    track (pd.DataFrame): DataFrame containing track data for a single particle
    
    Returns:
    pd.DataFrame: Normalized track data
    """
    # Ensure the landing frame exists in the track data
    if track['frame_shifted'].min() > 0 or track['frame_shifted'].max() < 0:
        print(f"Warning: Landing frame (frame_shifted = 0) not found for particle {track['particle'].iloc[0]}")
        # Use the first frame as a fallback
        landing_frame = track['frame'].min()
    else:
        # Find the frame closest to the landing frame (where frame_shifted is closest to 0)
        landing_frame = track.loc[track['frame_shifted'].abs().idxmin(), 'frame']
    
    # Get values at landing frame
    MFI_GCaMP_at_landing = track.loc[track['frame'] == landing_frame, 'MFI_GCaMP_smooth'].values[0]
    MFI_RFP_at_landing = track.loc[track['frame'] == landing_frame, 'MFI_RFP_smooth'].values[0]
    ratio_at_landing = track.loc[track['frame'] == landing_frame, 'ratio_smooth'].values[0]
    
    # Normalize GCaMP
    if MFI_GCaMP_at_landing == 0:
        track['MFI_GCaMP_normalized'] = track['MFI_GCaMP_smooth']
    else:
        track['MFI_GCaMP_normalized'] = track['MFI_GCaMP_smooth'] / MFI_GCaMP_at_landing
    
    # Normalize RFP
    if MFI_RFP_at_landing == 0:
        track['MFI_RFP_normalized'] = track['MFI_RFP_smooth']
    else:
        track['MFI_RFP_normalized'] = track['MFI_RFP_smooth'] / MFI_RFP_at_landing
    
    # Normalize ratio
    if ratio_at_landing == 0:
        track['ratio_normalized'] = track['ratio_smooth']
    else:
        track['ratio_normalized'] = track['ratio_smooth'] / ratio_at_landing
    
    return track


def plot_normalized_mfi(tracks):
    """Plot the normalized MFI based on the landing frame"""
    # Create a figure with three subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    # Plot GCaMP
    for name, group in tracks.groupby('particle'):
        axs[0].plot(group['frame_shifted'], group['MFI_GCaMP_normalized'], alpha=0.3)
    axs[0].set_ylabel('Normalized GCaMP')
    axs[0].set_title('Normalized and Aligned GCaMP by Landing Frame')
    
    # Plot RFP
    for name, group in tracks.groupby('particle'):
        axs[1].plot(group['frame_shifted'], group['MFI_RFP_normalized'], alpha=0.3)
    axs[1].set_ylabel('Normalized RFP')
    axs[1].set_title('Normalized and Aligned RFP by Landing Frame')
    
    # Plot ratio
    for name, group in tracks.groupby('particle'):
        axs[2].plot(group['frame_shifted'], group['ratio_normalized'], alpha=0.3)
    axs[2].set_xlabel('Frame Shifted')
    axs[2].set_ylabel('Normalized Ratio (GCaMP/RFP)')
    axs[2].set_title('Normalized and Aligned Ratio by Landing Frame')
    
    plt.tight_layout()
    plt.show()
    
    # Also plot the mean of all tracks for each metric
    plt.figure(figsize=(12, 6))
    
    mean_GCaMP = tracks.groupby('frame_shifted')['MFI_GCaMP_normalized'].mean()
    mean_RFP = tracks.groupby('frame_shifted')['MFI_RFP_normalized'].mean()
    mean_ratio = tracks.groupby('frame_shifted')['ratio_normalized'].mean()
    
    plt.plot(mean_GCaMP.index, mean_GCaMP.values, label='GCaMP')
    plt.plot(mean_RFP.index, mean_RFP.values, label='RFP')
    plt.plot(mean_ratio.index, mean_ratio.values, label='Ratio')
    
    plt.xlabel('Frame Shifted')
    plt.ylabel('Mean Normalized Value')
    plt.title('Mean Normalized Values by Landing Frame')
    plt.legend()
    plt.show()

# Debugging function
def print_normalization_check(tracks):
    """Print MFI_normalized values at frame_shifted 0 for each particle"""
    for name, group in tracks.groupby('particle'):
        print(f"Particle {name}: MFI_normalized at frame_shifted 0: {group.loc[group['frame_shifted'] == 0, 'MFI_normalized'].values}")


def smooth_and_normalize_tracks(tracks, gcamp_bleach_params, rfp_bleach_params):
    """
    Smooth MFI data, detect landing frames, and normalize tracks.
    Now applies photobleaching correction based on automatically detected landing frames,
    before interactive selection.
    
    Args:
    tracks (pd.DataFrame): DataFrame containing track data
    gcamp_bleach_params (tuple): Parameters for GCaMP photobleaching
    rfp_bleach_params (tuple): Parameters for RFP photobleaching
    
    Returns:
    pd.DataFrame: Processed tracks with smoothing, photobleaching correction, and normalization
    """
    background = 6.0
    # Subtract background
    tracks['MFI_GCaMP'] = tracks['MFI_GCaMP'] - 6.5
    # Ensure no negative values after background subtraction
    tracks.loc[tracks['MFI_GCaMP'] < 0, 'MFI_GCaMP'] = 0

    tracks['MFI_RFP'] = tracks['MFI_RFP'] - 0.0
    # Ensure no negative values after background subtraction
    tracks.loc[tracks['MFI_RFP'] < 0, 'MFI_RFP'] = 0
    
    # Recalculate ratio after background subtraction
    tracks['ratio'] = tracks['MFI_GCaMP'] / tracks['MFI_RFP']

    # Smooth all three metrics
    tracks['MFI_GCaMP_smooth'] = tracks.groupby('particle')['MFI_GCaMP'].transform(smooth_track)
    tracks['MFI_RFP_smooth'] = tracks.groupby('particle')['MFI_RFP'].transform(smooth_track)
    tracks['ratio_smooth'] = tracks.groupby('particle')['ratio'].transform(smooth_track)
    
    # Auto-detect landing frames
    print('Detecting landing frames...')
    landing_frames = tracks.groupby('particle').apply(detect_landing_frame)
    tracks['landing_frame'] = tracks['particle'].map(landing_frames)
    
    # Temporarily shift frames based on auto-detected landing frames for photobleaching correction
    tracks['frame_shifted'] = tracks['frame'] - tracks['landing_frame']
    
    # Apply photobleaching correction after automatic landing frame detection
    # This ensures correction starts from the automatically detected landing frame
    tracks = correct_photobleaching(tracks, gcamp_bleach_params, rfp_bleach_params)
    
    # Interactive landing frame selection and track filtering
    updated_landing_frames = interactive_landing_frame_selection(tracks)
    
    # Filter out discarded tracks and update landing frames
    tracks = tracks[tracks['particle'].isin(updated_landing_frames.keys())].copy()
    
    # Update landing frames with manually selected or confirmed frames
    tracks['landing_frame'] = tracks['particle'].map(updated_landing_frames)
    
    # Re-compute frame_shifted based on potentially updated landing frames
    tracks['frame_shifted'] = tracks['frame'] - tracks['landing_frame']
    
    # Re-smooth data after photobleaching correction
    tracks['MFI_GCaMP_smooth'] = tracks.groupby('particle')['MFI_GCaMP'].transform(smooth_track)
    tracks['MFI_RFP_smooth'] = tracks.groupby('particle')['MFI_RFP'].transform(smooth_track)
    tracks['ratio_smooth'] = tracks.groupby('particle')['ratio'].transform(smooth_track)
    
    # Normalize tracks
    print('Normalizing tracks...')
    tracks = tracks.groupby('particle').apply(normalize_by_landing).reset_index(drop=True)
    
    return tracks


def interactive_landing_frame_selection(tracks):
    """
    Allow manual selection of landing frames for each particle with validation.
    Includes options to confirm auto-detection or discard problematic tracks.
    
    Args:
    tracks (pd.DataFrame): DataFrame containing track data for all particles
    
    Returns:
    dict: Updated landing frames for each particle
    """
    updated_landing_frames = {}
    
    # Get all unique particles as a list to enable navigation
    particles = sorted(tracks['particle'].unique())
    particle_idx = 0
    
    while particle_idx < len(particles):
        particle = particles[particle_idx]
        group = tracks[tracks['particle'] == particle].copy()
        
        # Create figure with proper layout
        fig = plt.figure(figsize=(10, 15))
        
        # Create layout with 3 plot axes and a small area at bottom for buttons
        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 3, 3, 0.5])
        
        axs = [plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2])]
        
        # Plot GCaMP MFI
        axs[0].plot(group['frame'], group['MFI_GCaMP'], label='GCaMP')
        axs[0].set_ylabel('MFI_GCaMP')
        axs[0].set_title(f'Particle {particle} ({particle_idx+1}/{len(particles)}): Use arrow keys or click to move, Enter to set frame')
        axs[0].legend()
        
        # Plot RFP MFI
        axs[1].plot(group['frame'], group['MFI_RFP'], label='RFP')
        axs[1].set_ylabel('MFI_RFP')
        axs[1].legend()
        
        # Plot ratio
        axs[2].plot(group['frame'], group['ratio'], label='GCaMP/RFP Ratio')
        axs[2].set_ylabel('Ratio')
        axs[2].set_xlabel('Frame')
        axs[2].legend()
        
        # Link x-axes for synchronized zooming/panning
        axs[1].sharex(axs[0])
        axs[2].sharex(axs[0])
        
        # Get available frames for this particle
        available_frames = sorted(group['frame'].unique())
        
        # Set initial landing frame to the auto-detected one
        auto_landing_frame = group['landing_frame'].iloc[0]
        
        # If frame is not in available frames, use the closest one
        if auto_landing_frame not in available_frames:
            current_frame = min(available_frames, key=lambda x: abs(x - auto_landing_frame))
        else:
            current_frame = auto_landing_frame
        
        # Find the appropriate index for the current frame
        frame_idx = available_frames.index(current_frame)
        
        # Crosshair lines (vertical and horizontal)
        vlines = []
        hlines = []
        for ax_idx, ax in enumerate(axs):
            vline = ax.axvline(x=current_frame, color='r', linestyle='--')
            vlines.append(vline)
            
            # Get y-value for this frame
            y_values = group.loc[group['frame'] == current_frame]
            
            if len(y_values) > 0:
                if ax_idx == 0:
                    y_val = y_values['MFI_GCaMP'].values[0]
                elif ax_idx == 1:
                    y_val = y_values['MFI_RFP'].values[0]
                else:
                    y_val = y_values['ratio'].values[0]
                    
                hline = ax.axhline(y=y_val, color='r', linestyle='--')
                hlines.append(hline)
            else:
                # No data at this frame, just add a placeholder horizontal line
                hline = ax.axhline(y=0, color='r', linestyle='--', alpha=0)
                hlines.append(hline)
        
        # Status text at the bottom
        status_text = plt.figtext(0.5, 0.01, f"Current frame: {current_frame}", ha="center", fontsize=10, 
                               bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        # Function to update crosshair position
        def update_crosshair(frame):
            nonlocal current_frame, frame_idx, status_text
            
            # Update the current frame and index
            current_frame = frame
            frame_idx = available_frames.index(frame)
            
            # Update all vertical lines
            for i, ax in enumerate(axs):
                vlines[i].set_xdata(frame)
                
                # Update horizontal line position
                y_values = group.loc[group['frame'] == frame]
                
                if len(y_values) > 0:
                    if i == 0:
                        y_val = y_values['MFI_GCaMP'].values[0]
                    elif i == 1:
                        y_val = y_values['MFI_RFP'].values[0]
                    else:
                        y_val = y_values['ratio'].values[0]
                        
                    hlines[i].set_ydata(y_val)
                    hlines[i].set_alpha(1)
                else:
                    hlines[i].set_alpha(0)
            
            # Update status text
            status_text.set_text(f"Current frame: {frame}")
            
            fig.canvas.draw_idle()
        
        # Create button axes with appropriate sizes
        ax_prev = plt.axes([0.1, 0.05, 0.15, 0.05])
        ax_next = plt.axes([0.27, 0.05, 0.15, 0.05])
        ax_set = plt.axes([0.44, 0.05, 0.15, 0.05])
        ax_confirm = plt.axes([0.61, 0.05, 0.15, 0.05])
        ax_discard = plt.axes([0.78, 0.05, 0.15, 0.05])
        
        # Create buttons
        button_prev = Button(ax_prev, '← Previous', color='lightblue', hovercolor='skyblue')
        button_next = Button(ax_next, 'Next →', color='lightblue', hovercolor='skyblue')
        button_set = Button(ax_set, 'Set Landing', color='lightgreen', hovercolor='lime')
        button_confirm = Button(ax_confirm, 'Confirm Auto', color='lightyellow', hovercolor='yellow')
        button_discard = Button(ax_discard, 'Discard Track', color='salmon', hovercolor='red')
        
        # Button click handlers
        def handle_prev(event):
            nonlocal particle_idx
            if particle_idx > 0:
                particle_idx -= 1
                plt.close(fig)
        
        def handle_next(event):
            nonlocal particle_idx
            if particle_idx < len(particles) - 1:
                particle_idx += 1
                plt.close(fig)
        
        def handle_set(event):
            updated_landing_frames[particle] = current_frame
            nonlocal particle_idx
            if particle_idx < len(particles) - 1:
                particle_idx += 1
            else:
                # Important fix: when we're at the last particle, set idx beyond the array length
                # to exit the while loop
                particle_idx = len(particles)
            plt.close(fig)
        
        def handle_confirm(event):
            updated_landing_frames[particle] = auto_landing_frame
            nonlocal particle_idx
            if particle_idx < len(particles) - 1:
                particle_idx += 1
            else:
                # Important fix: when we're at the last particle, set idx beyond the array length
                # to exit the while loop
                particle_idx = len(particles)
            plt.close(fig)
        
        def handle_discard(event):
            # Don't add to updated_landing_frames
            nonlocal particle_idx
            if particle_idx < len(particles) - 1:
                particle_idx += 1
            else:
                # Important fix: when we're at the last particle, set idx beyond the array length
                # to exit the while loop
                particle_idx = len(particles)
            plt.close(fig)
        
        # Function to handle mouse clicks on the plot
        def on_plot_click(event):
            # Only process clicks inside the plot axes
            if event.inaxes in axs:
                # Get the x-coordinate (frame) of the click
                clicked_frame = event.xdata
                
                # Find the closest available frame to where we clicked
                closest_frame = min(available_frames, key=lambda x: abs(x - clicked_frame))
                
                # Update the crosshair to the new position
                update_crosshair(closest_frame)
        
        # Connect button event handlers
        button_prev.on_clicked(handle_prev)
        button_next.on_clicked(handle_next)
        button_set.on_clicked(handle_set)
        button_confirm.on_clicked(handle_confirm)
        button_discard.on_clicked(handle_discard)
        
        # Function to handle key press events
        def on_key_press(event):
            nonlocal frame_idx, current_frame
            
            if event.key == 'right' and frame_idx < len(available_frames) - 1:
                frame_idx += 1
                current_frame = available_frames[frame_idx]
                update_crosshair(current_frame)
                
            elif event.key == 'left' and frame_idx > 0:
                frame_idx -= 1
                current_frame = available_frames[frame_idx]
                update_crosshair(current_frame)
                
            elif event.key == 'enter':
                updated_landing_frames[particle] = current_frame
                nonlocal particle_idx
                if particle_idx < len(particles) - 1:
                    particle_idx += 1
                else:
                    # Important fix: when we're at the last particle, set idx beyond the array length
                    # to exit the while loop
                    particle_idx = len(particles) 
                plt.close(fig)
        
        # Connect the key press event and the mouse click event
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.mpl_connect('button_press_event', on_plot_click)
        
        # Show whether this is the last particle
        if particle_idx == len(particles) - 1:
            plt.figtext(0.5, 0.95, "LAST TRACK", ha="center", fontsize=12, 
                       color="red", weight="bold")
        
        # Adjust spacing between subplots
        plt.tight_layout(rect=[0, 0.12, 1, 0.98])  # Adjust layout but leave room for buttons
        
        # Make sure figure has focus for keyboard events
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        plt.show()
        
        # If no selection was made, use auto-detected landing frame
        if particle not in updated_landing_frames:
            updated_landing_frames[particle] = auto_landing_frame
    
    return updated_landing_frames