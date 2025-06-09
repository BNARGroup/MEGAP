import mne
import os
from initial_files import initial_files
from initial_py import read_cfg
from mne_bids import BIDSPath, write_raw_bids

def write_bids(subject_ids):
    """
    Convert MEG data into BIDS format and write it to disk.

    The function reads preprocessed MEG data for each subject and writes the data in BIDS format.
    """
    # Get the main directory for MEG data from environment variables
    main_location = str(os.getenv('MAINMEG'))

    # Read the configuration settings for writing to BIDS
    cfg = read_cfg(main_location)
    step = 'BIDS_structure_write'
    cfg = cfg[step]

    # Loop through each subject
    for subject_id in subject_ids:
        # Load the MEG data for the current subject
        raw, _, folder_path = initial_files(subject_id, main_location, "preprocessed_BIDS_output", "ICA")

        # Set cHPI channels to 'misc' type (if any exist)
        chpi_channels = {ch_name: 'misc' for ch_name in raw.ch_names if ch_name.startswith('CHPI')}
        if chpi_channels:
            raw.set_channel_types(chpi_channels)  

        # Modify the subject ID to match BIDS format 
        subject_id_bids = subject_id[4:]

        # Define the BIDS path using the BIDSPath function and configuration
        bids_path = BIDSPath(subject=subject_id_bids, root=folder_path, processing='ICA', **cfg)

        # Write the raw data in BIDS format
        write_raw_bids(raw, bids_path=bids_path, verbose=False, allow_preload=True, format="FIF", overwrite=True)
        print(f"Data written to BIDS for subject: {subject_id}")
