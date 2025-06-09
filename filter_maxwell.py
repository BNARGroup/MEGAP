# Import necessary libraries
import os
import mne
from initial_files import initial_files
from initial_py import read_cfg, redirect_stdout

def filter_maxwell(subject_ids):
    """
    Apply Maxwell filtering to MEG data and save the filtered data for each subject.

    This function reads configuration files, applies Maxwell filtering to the MEG data to remove
    noise and interference, and saves the filtered data in FIF format.
    """
    # Get the main directory location from an environment variable
    main_location = str(os.getenv('MAINMEG'))
    
    # Read the configuration settings for the 'filter_maxwell' step
    cfg = read_cfg(main_location)
    step = 'filter_maxwell'
    cfg = cfg[step]

    # Loop over each subject
    for subject_id in subject_ids:
        # Create a log file to store verbose output for each subject
        log_file_path = os.path.join(main_location, 'verbose', f'{subject_id}.txt')
        with open(log_file_path, 'a') as log_file:
            with redirect_stdout(log_file, step):
                print(f"Processing subject: {subject_id}")
                print(f"Configuration used for Maxwell filtering: {cfg}\n")

                # Load the MEG data and initialize folder structure
                data, head_pos, folder_path = initial_files(subject_id, main_location, "filter_maxwell", "OTP")

                # Apply Maxwell filtering to the raw MEG data
                data_sss = mne.preprocessing.maxwell_filter(
                    data,
                    **cfg,
                    calibration=os.path.join(main_location, "config", "sss_cal.dat"),  # Path to calibration file
                    cross_talk=os.path.join(main_location, "config", "ct_sparse.fif"),  # Path to cross-talk file
                    head_pos=head_pos,
                    skip_by_annotation=('BAD_muscle')  # Skip bad muscle
                )

                # Interpolate bad channels using minimum-norm method
                data_sss.interpolate_bads(reset_bads=False)

                # Save the filtered data in FIF format
                filtered_data_path = os.path.join(folder_path, f"{subject_id}.fif")
                data_sss.save(filtered_data_path, overwrite=True, fmt='double')
                print(f"Filtered Maxwell data saved for subject {subject_id} at {filtered_data_path}\n")
