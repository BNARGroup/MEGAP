# Import necessary libraries
import os
import pickle
import numpy as np
import mne
import datetime
from initial_files import initial_files
from initial_py import read_cfg, redirect_stdout
from initial_py import find_meg_device

def filter_chpi(subject_ids):
    """
    Filter continuous head position indicator (cHPI) noise from MEG data for each subject.
 
    This function loads the raw MEG data, applies the cHPI filter, and saves the filtered data in FIF format.
    """
    # Get the main directory location from an environment variable
    main_location = str(os.getenv('MAINMEG'))
    MEG_device = str(os.getenv('MEGDEVICE'))
    find_meg_device()
    # Read the configuration for the 'filter_chpi' step
    cfg = read_cfg(main_location)
    step = 'filter_chpi'
    cfg_filter = cfg[step]
    
    # Loop over each subject
    for subject_id in subject_ids:
        # Create a log file to store verbose output for each subject
        log_file_path = os.path.join(main_location, 'verbose', f'{subject_id}.txt')
        with open(log_file_path, 'a') as log_file:
            with redirect_stdout(log_file, step):
                print(f"Processing subject: {subject_id}")

                # Load the MEG data and initialize folder structure
                data, _, folder_path = initial_files(subject_id, main_location, "filter_chpi", "data")

                # Load previously detected flat channels
                folder_path_bad_channel = os.path.join(main_location, "flat_channel")
                with open(os.path.join(folder_path_bad_channel, f'{subject_id}.pkl'), 'rb') as file:
                    auto_flat_chs = pickle.load(file)
                data.info['bads'] = auto_flat_chs  # Mark flat channels as bad

                if data.info["hpi_results"]:
                    if MEG_device=="MEGIN":
                        print(f"Configuration used for filtering: {cfg_filter}\n")
                        # Apply cHPI filtering to the raw MEG data
                        data_chpi_filt = mne.chpi.filter_chpi(data, **cfg_filter)

                        # Save the filtered data in FIF format
                        filtered_data_path = os.path.join(folder_path, f"{subject_id}.fif")
                        data_chpi_filt.save(filtered_data_path, overwrite=True, fmt='double')
                        print(f"Filtered data saved for subject {subject_id} at {filtered_data_path}\n")
                    else:
                        print("CTF data do not need cHPI filtering as this MEG system dont have any cHPI pick on its PSD")
                else:
                    print("Could not find any cHPI data")
                current_time = datetime.datetime.now()
                print("Completation time:", current_time.strftime("%Y-%m-%d %H:%M:%S"), "\n")

def chpi_crop(subject_ids):
    """
    Crop the MEG data to exclude the initial period before cHPI (continuous head position indicator) signals are turned on.

    This function detects the time when cHPI turns on and crops the data to remove the portion before this time.
    """
    # Get the main directory location from an environment variable
    main_location = str(os.getenv('MAINMEG'))
    MEG_device = str(os.getenv('MEGDEVICE'))
    step = "chpi_crop"

    # Loop over each subject
    for subject_id in subject_ids:
        # Create a log file to store verbose output for each subject
        log_file_path = os.path.join(main_location, 'verbose', f'{subject_id}.txt')
        with open(log_file_path, 'a') as log_file:
            with redirect_stdout(log_file, step):
                print(f"Processing subject: {subject_id}")

                # Load the raw MEG data
                data, _, folder_path = initial_files(subject_id, main_location, "data", "raw")
                sfreq = data.info['sfreq']  # Sampling frequency
                start_chpi=0 # Instead of MEGIN, Other MEG devices have a value=0 in cHPI initialization time

                if data.info["hpi_results"]:

                    if MEG_device=="MEGIN":
                        # Get cHPI info; if not available, pick system-related channels
                        _, ch_idx, _ = mne.chpi.get_chpi_info(data.info)
                        if ch_idx is None:
                            ch_idx = mne.pick_types(data.info, syst=True)
                            ch_idx = ch_idx[0]
                        stim = data.get_data(picks=ch_idx)
                        stim_diff = np.diff(stim)
                        # Find the time when the cHPI turns on
                        start_chpi = (np.array(stim_diff).argmax() / sfreq)
                        
                    elif MEG_device=="CTF":
                        data.apply_gradient_compensation(0) # Gradient compensation will be applied at the end of pipeline

                    total_start = start_chpi + data.first_time
                    # Print diagnostic information
                    print(f"cHPI turn on detected at: {start_chpi} seconds")
                    print(f"First time in the dataset: {data.first_time} seconds")
                    print(f"New start point in total time: {total_start} seconds")
                    print(f"Length of the data: {data.n_times / sfreq} seconds")
                else:
                    print("Could not find any cHPI data")

                total_start = start_chpi + data.first_time

                # Crop the data to exclude the portion before the cHPI starts
                # Without cHPI data, this function just save the data in .fif format (e.g. CTF or BDI)
                # data.crop(tmin=start_chpi,tmax=start_chpi+70)#include_tmax=False)

                # Save the cropped data in FIF format
                cropped_data_path = os.path.join(folder_path, f"{subject_id}.fif")
                data.save(cropped_data_path, overwrite=True, fmt='double')
                print(f"Cropped data saved for subject {subject_id} at {cropped_data_path}\n")

                current_time = datetime.datetime.now()
                print("Completation time:", current_time.strftime("%Y-%m-%d %H:%M:%S"), "\n")