# Import necessary libraries
import os
import pickle
import numpy as np
import mne
import datetime
from initial_files import initial_files
from initial_py import read_cfg, redirect_stdout, write_warning
from scipy.stats import median_abs_deviation
from scipy import signal
from scipy.stats import median_abs_deviation
from autoreject import Ransac
from mne.preprocessing import create_ecg_epochs
from initial_py import find_meg_device

def bad_channel_detection(subject_ids):
    """
    Detect and save bad channels in MEG data for a list of subjects.

    """
    # Get the main directory location from an environment variable
    main_location = str(os.getenv('MAINMEG'))
    MEG_device = str(os.getenv('MEGDEVICE'))
    find_meg_device()
    # Read the configuration for 'bad_channel' step
    cfg = read_cfg(main_location)
    step = 'bad_channel_maxwell'
    cfg = cfg[step]
    
    cfg_nonmaxwell = read_cfg(main_location)
    step_nonmaxwell = 'bad_channel_nonmaxwell'
    cfg_nonmaxwell = cfg_nonmaxwell[step_nonmaxwell]

    # Loop over each subject
    for subject_id in subject_ids:
        # Create a log file to store verbose output for each subject
        log_file_path = os.path.join(main_location, 'verbose', f'{subject_id}.txt')
        with open(log_file_path, 'a') as log_file:
            with redirect_stdout(log_file, step):
                print(f"Config for subject {subject_id}: {cfg}\n")

                # Load the MEG data and initialize folder structure
                data, head_pos, folder_path = initial_files(subject_id, main_location, "bad_channel", "data")

                if MEG_device=="MEGIN":
                # Automatically detect bad channels 
                    auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(
                        data,
                        **cfg,
                        head_pos=head_pos,
                        calibration=os.path.join(main_location, "config", "sss_cal.dat"),  # Path to calibration file
                        cross_talk=os.path.join(main_location, "config", "ct_sparse.fif"),  # Path to cross-talk file
                        verbose=True
                    )
                    results = (auto_noisy_chs, auto_flat_chs, auto_scores)

                    all_bad_channels = auto_noisy_chs + auto_flat_chs 

                else:
                    all_bad_channels,auto_noisy_chs,auto_flat_chs,auto_scores,bad_by_type=find_bad_channels_non_maxwell(data,**cfg_nonmaxwell)
                    results = (auto_noisy_chs, auto_flat_chs, auto_scores)
                    
                    # Print results
                    print("Total bad channels found:", len(all_bad_channels))
                    print("\nBad channels by type:")
                    for detection_type, channels in bad_by_type.items():
                        print(f"\n{detection_type}:", channels)

                # Save the detected bad channels and scores to a pickle file
                output_path = os.path.join(folder_path, f'{subject_id}.pkl')
                with open(output_path, 'wb') as file:
                    pickle.dump(results, file)
                print(f"Bad channels for {subject_id} saved to {output_path}\n")

        # Use a set to remove duplicates, then convert back to a list
        all_bad_channels = list(set(all_bad_channels))
        total_channels = mne.pick_types(data.info, meg=True, ref_meg=False)
        write_bad_channel_warning(subject_id, all_bad_channels, total_channels)

def find_bad_channels_non_maxwell(data,deviation_thresh=5.0, correlation_thresh=0.4, 
                         hf_thresh=5.0, snr_enabled=True):

    auto_noisy_chs=[]
    auto_flat_chs=[]
    auto_scores=[]


    meg_picks_diff =  mne.pick_types(data.info, meg='mag', ref_meg=False)
    meg_data = data.get_data(picks=meg_picks_diff)
    ch_names = [data.ch_names[i] for i in meg_picks_diff]

    # Filter data for HF noise and SNR detection
    filtered_data = filter_meg_data(meg_data, data.info['sfreq'])
    sfreq=data.info['sfreq']
    # Run each detection method
    bad_by_nan_flat = find_bad_by_nan_flat(meg_data, ch_names)
    bad_by_deviation = find_bad_by_deviation(meg_data, ch_names, deviation_thresh)
    bad_by_correlation = find_bad_by_correlation(
        meg_data, ch_names, 
        sfreq,
        correlation_thresh,
    )
    bad_by_hf = find_bad_by_hf_noise(meg_data, ch_names, filtered_data, hf_thresh)
    bad_by_ransac=[]
    try:
        bad_by_ransac=find_bad_by_ransac(data)
    except: 
        print("Can not run ransac for bad channel detection. Check your digitization points in your data")

    # SNR detection (optional)
    bad_by_snr = []
    if snr_enabled:
        bad_by_snr = find_bad_by_SNR(
            meg_data, ch_names, filtered_data,sfreq,
            correlation_thresh, hf_thresh
        )
    
    # Combine all bad channels
    all_bad_channels = list(set(
        bad_by_ransac +
        bad_by_nan_flat + 
        bad_by_deviation + 
        bad_by_correlation +
        bad_by_hf +
        bad_by_snr
    ))
    auto_noisy_chs = list(set(
        bad_by_ransac +
        bad_by_nan_flat + 
        bad_by_deviation + 
        bad_by_correlation +
        bad_by_hf +
        bad_by_snr
    ))
    auto_flat_chs = list(set(
        bad_by_nan_flat 
    ))
    
    # Create dictionary of results by type
    bad_by_type = {
        'bad_by_ransac': bad_by_ransac,
        'bad_by_nan_flat': bad_by_nan_flat,
        'bad_by_deviation': bad_by_deviation,
        'bad_by_correlation': bad_by_correlation,
        'bad_by_hf_noise': bad_by_hf,
        'bad_by_snr': bad_by_snr,
        'all_bad_channels': all_bad_channels
    }
    
    return all_bad_channels,auto_noisy_chs,auto_flat_chs,auto_scores,bad_by_type



def find_flat_channel(matrix, meg_ch_idx,flat_threshold=1e-18):
    """
    Args:
        matrix (2D array): MEG signal matrix for each channel.
        meg_ch_idx (list): List of indices corresponding to MEG channels.

    """
    # Convert the input matrix to a NumPy array
    matrix = np.array(matrix)

    # Initialize list to store indices of flat channels
    flat_channel_indices = []

    # Return -1 if the matrix is empty (no channels)
    if matrix.size == 0:
        return -1

    # Check if the matrix is 2D
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be 2D")

    # Detect channels containing any NaN values
    nan_channel_mask = np.isnan(np.sum(matrix, axis=1))
    # nan_channels = np.array(ch_names)[nan_channel_mask]
    
    # Detect channels with flat signals
    flat_by_mad = median_abs_deviation(matrix, axis=1) < flat_threshold
    flat_by_std = np.std(matrix, axis=1) < flat_threshold
    flat_channel_mask = flat_by_mad | flat_by_std

    flat_channel_indices.extend(meg_ch_idx[flat_channel_mask])
    flat_channel_indices.extend(meg_ch_idx[nan_channel_mask])

    return flat_channel_indices


def flat_channel(subject_ids):
    """
    Detect and save flat channels.

    """ 
    main_location = str(os.getenv('MAINMEG'))
    MEG_device = str(os.getenv('MEGDEVICE'))
    find_meg_device()
    step = "flat_channel"
    # Loop over each subject
    for subject_id in subject_ids:
        log_file_path = os.path.join(main_location, 'verbose', f'{subject_id}.txt')
        with open(log_file_path, 'a') as log_file:
            with redirect_stdout(log_file, step):
                print(f"Processing subject: {subject_id}")
                # Load the MEG data and initialize folder structure
                data, _, folder_path = initial_files(subject_id, main_location, "flat_channel", "data")


                # Pick the MEG channel indices
                meg_ch_idx = mne.pick_types(data.info, meg=True , ref_meg=False)

                # Get the signal data for the MEG channels
                signals = data.get_data(picks=meg_ch_idx)
                
                # Find the flat channels
                flat_channel_indices = find_flat_channel(signals, meg_ch_idx)
                flat_channel_indices=[1]
                ch_names = data.info['ch_names']
                
                # Get the channel names of the flat channels
                auto_flat_chs = [ch_names[idx] for idx in flat_channel_indices]


                # Print and save flat channel names
                print(f"Flat channels for {subject_id}: {auto_flat_chs}")
                output_path = os.path.join(folder_path, f'{subject_id}.pkl')
                with open(output_path, 'wb') as file:
                    pickle.dump(auto_flat_chs, file)

                current_time = datetime.datetime.now()
                print("Completation time:", current_time.strftime("%Y-%m-%d %H:%M:%S"), "\n")


def find_bad_by_nan_flat(meg_data, ch_names, flat_threshold=1e-18):
    """Detect MEG channels with NaN values or flat signals.
    
    Parameters
    ----------
    meg_data : np.ndarray
        MEG data array of shape (n_channels, n_samples)
    ch_names : list
        List of channel names
    flat_threshold : float 
        Threshold for detecting flat sensors
        
    Returns
    -------
    bad_channels : list
        Names of detected bad channels
    """
    # Detect channels containing any NaN values
    nan_channel_mask = np.isnan(np.sum(meg_data, axis=1))
    nan_channels = np.array(ch_names)[nan_channel_mask]
    
    # Detect channels with flat signals
    flat_by_mad = median_abs_deviation(meg_data, axis=1) < flat_threshold
    flat_by_std = np.std(meg_data, axis=1) < flat_threshold
    flat_channel_mask = flat_by_mad | flat_by_std
    flat_channels = np.array(ch_names)[flat_channel_mask]
    
    return list(nan_channels) + list(flat_channels)

def find_bad_by_ransac (raw_data):

    eog_evoked = create_ecg_epochs(raw_data)
    eog_evoked.apply_baseline(baseline=(None, -0.2))


    picks = mne.pick_types(raw_data.info, meg=True, eeg=False,
                        stim=False, eog=False, ref_meg=False,
                        include=[], exclude=[])

    ransac = Ransac(verbose=True, picks=picks, n_jobs=1)
    epochs_clean = ransac.fit_transform(eog_evoked)

    return   ransac.bad_chs_

def find_bad_by_deviation(meg_data, ch_names, deviation_threshold=5.0):
    """Detect MEG channels with abnormal amplitudes.
    
    Parameters
    ----------
    meg_data : np.ndarray
        MEG data array of shape (n_channels, n_samples)
    ch_names : list
        List of channel names
    deviation_threshold : float
        Z-score threshold for detecting deviant channels
        
    Returns
    -------
    bad_channels : list
        Names of detected bad channels
    """
    # Calculate robust channel amplitudes
    chan_amplitudes = np.std(meg_data, axis=1)
    amp_median = np.median(chan_amplitudes)
    amp_mad = median_abs_deviation(chan_amplitudes)
    
    # Calculate z-scores
    z_scores = (chan_amplitudes - amp_median) / (amp_mad * 1.4826)
    
    # Identify bad channels
    bad_mask = np.abs(z_scores) > deviation_threshold
    return list(np.array(ch_names)[bad_mask])

def find_bad_by_correlation(meg_data, ch_names, sfreq,correlation_threshold=0.4, window_secs=1.0):
    """Detect MEG channels with poor correlation with other channels.
    
    Parameters
    ----------
    meg_data : np.ndarray
        MEG data array of shape (n_channels, n_samples)
    ch_names : list
        List of channel names
    correlation_threshold : float
        Minimum correlation threshold
    window_secs : float
        Window length in seconds
    sfreq : float
        Sampling frequency
        
    Returns
    -------
    bad_channels : list
        Names of detected bad channels
    """
    window_size = int(window_secs * sfreq)
    n_windows = meg_data.shape[1] // window_size
    
    bad_windows = np.zeros((len(ch_names),))
    
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        data_window = meg_data[:, start:end]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(data_window)
        
        # Get maximum correlation for each channel (excluding self-correlation)
        np.fill_diagonal(corr_matrix, 0)
        max_corrs = np.max(np.abs(corr_matrix), axis=1)
        
        bad_windows += (max_corrs < correlation_threshold)
    
    # Mark channels as bad if they're bad in > 1% of windows
    bad_mask = bad_windows / n_windows > 0.01
    return list(np.array(ch_names)[bad_mask])


def find_bad_by_hf_noise(meg_data, ch_names, filtered_data, zscore_threshold=5.0):
    """Detect MEG channels with high frequency noise.
    
    Parameters
    ----------
    meg_data : np.ndarray
        Raw MEG data array of shape (n_channels, n_samples)
    ch_names : list
        List of channel names
    filtered_data : np.ndarray
        Filtered MEG data array of shape (n_channels, n_samples)
    zscore_threshold : float
        Z-score threshold for detecting noisy channels
        
    Returns 
    -------
    bad_channels : list
        Names of detected bad channels
    """
    # Calculate ratio of HF to total signal
    noise = meg_data - filtered_data
    noise_ratio = median_abs_deviation(noise, axis=1) / median_abs_deviation(filtered_data, axis=1)
    
    # Calculate z-scores of noise ratios
    noise_median = np.median(noise_ratio)
    noise_mad = median_abs_deviation(noise_ratio)
    noise_zscores = (noise_ratio - noise_median) / (noise_mad * 1.4826)
    
    # Identify bad channels
    bad_mask = noise_zscores > zscore_threshold
    return list(np.array(ch_names)[bad_mask])

def find_bad_by_SNR(meg_data, ch_names, filtered_data,sfreq, correlation_threshold=0.4, hf_threshold=5.0):
    """Detect MEG channels with poor SNR (both noisy and poorly correlated).
    
    Parameters
    ----------
    meg_data : np.ndarray
        MEG data array of shape (n_channels, n_samples)
    ch_names : list
        List of channel names
    filtered_data : np.ndarray
        Filtered MEG data array
    correlation_threshold : float
        Threshold for correlation-based detection
    hf_threshold : float
        Threshold for high-frequency noise detection
        
    Returns
    -------
    bad_channels : list
        Names of channels with poor SNR
    """
    # Get channels bad by correlation
    bad_corr = set(find_bad_by_correlation(meg_data, ch_names,sfreq, correlation_threshold))
    
    # Get channels bad by high frequency noise
    bad_hf = set(find_bad_by_hf_noise(meg_data, ch_names, filtered_data, hf_threshold))
    
    # Channels bad by both criteria are considered to have poor SNR
    bad_snr = list(bad_corr.intersection(bad_hf))
    return bad_snr

def filter_meg_data(meg_data, sfreq, l_freq=1.0, h_freq=50.0):
    """Apply bandpass filter to MEG data.
    
    Parameters
    ----------
    meg_data : np.ndarray
        MEG data array of shape (n_channels, n_samples)
    sfreq : float
        Sampling frequency
    l_freq : float
        Lower frequency bound
    h_freq : float
        Upper frequency bound
        
    Returns
    -------
    filtered_data : np.ndarray
        Filtered MEG data
    """
    filtered_data = mne.filter.filter_data(
        meg_data,
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method='fir',
        fir_design='firwin'
    )
    return filtered_data
 
def write_bad_channel_warning(subject_id, bad_channels, total_channels):

    """
    Check if the number of bad channels exceeds a specified threshold count
    and write a warning message to a file if it does.
    
    Parameters:
    - subject_id: ID of the subject being processed
    - bad_channels: List of bad channels
    - total_channels: Total number of channels in the dataset
    - threshold_count: Threshold count for bad channels
    """
    main_location = str(os.getenv('MAINMEG'))
    cfg = read_cfg(main_location)
    cfg_warning = cfg['warning']
    limit_bad_channel=cfg_warning['bad_channel']

    # Calculate the number of bad channels
    num_bad_channels = len(bad_channels)

    # Check if the number of bad channels exceeds the threshold
    if num_bad_channels > limit_bad_channel:
        # Create the warning message
        warning_message = ("\n"+"_"*10+"Bad Channel"+"_"*10+"\n"+
            f"Warning: {num_bad_channels} bad channels detected, exceeding the threshold of {limit_bad_channel} bad channels.\n"
            f"There are {num_bad_channels} bad channels out of {len(total_channels)} total channels in the dataset.\n"
            "Please check the bad channels and consider reprocessing or correcting the data."
        )
        write_warning(subject_id,warning_message)