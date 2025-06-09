# Import necessary libraries
import os
import pickle
import mne
from initial_files import initial_files
from initial_py import read_cfg, redirect_stdout,find_meg_device

def OTP(subject_ids):
    """
    Apply Oversampled Temporal Projection (OTP) filtering to MEG data and save the processed data.

    This function performs the following:
    - Loads MEG data
    - Applies bad channel information and annotations
    - Applies OTP filtering to the MEG data
    - Saves the processed MEG data in FIF format

    """
    # Get the main directory of the MEG data from environment variables
    main_location = str(os.getenv('MAINMEG'))
    MEG_device = str(os.getenv('MEGDEVICE'))
    find_meg_device()
    cfg = read_cfg(main_location)
    step = 'OTP'  
    cfg = cfg[step]

    if MEG_device=="MEGIN":
        # Loop through each subject in the list of subject IDs
        for subject_id in subject_ids:
            # Create a log file to capture the process for the current subject
            log_file_path = os.path.join(main_location, 'verbose', f'{subject_id}.txt')
            with open(log_file_path, 'a') as log_file:
                with redirect_stdout(log_file, step):
                    print(f"Processing OTP for subject: {subject_id}")

                    # Load the MEG data using initial_files function
                    data, _, folder_path = initial_files(subject_id, main_location, "OTP", "multi_taper_removal")

                    # Load bad channel information
                    # Flat channel information
                    flat_channel_path = os.path.join(main_location, "flat_channel")
                    with open(os.path.join(flat_channel_path, f'{subject_id}.pkl'), 'rb') as file:
                        flat_channels = pickle.load(file)

                    # Auto-detected noisy and flat channels
                    bad_channel_path = os.path.join(main_location, "bad_channel")
                    with open(os.path.join(bad_channel_path, f'{subject_id}.pkl'), 'rb') as file:
                        auto_noisy_chs, auto_flat_chs, _ = pickle.load(file)

                    # Combine all bad channels into the data's bad channel list
                    data.info['bads'] = auto_noisy_chs + auto_flat_chs + flat_channels

                    # Load muscle artifact annotations
                    muscle_annotation_path = os.path.join(main_location, "annotate_muscle")
                    with open(os.path.join(muscle_annotation_path, f'{subject_id}.pkl'), 'rb') as file:
                        muscle_annotations = pickle.load(file)

                    # Set the muscle artifact annotations in the data
                    data.set_annotations(muscle_annotations)

                    # Apply Oversampled Temporal Projection (OTP) filtering
                    data_otp = mne.preprocessing.oversampled_temporal_projection(data, **cfg)

                    # Save the filtered data in FIF format
                    output_file_path = os.path.join(folder_path, f"{subject_id}.fif")
                    data_otp.save(output_file_path, overwrite=True, fmt='double')
                    print(f"OTP filtering completed and saved for subject: {subject_id}\n")
