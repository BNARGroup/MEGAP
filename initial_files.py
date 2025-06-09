# Import necessary libraries
import os
import mne
from initial_py import read_cfg
from mne_bids import BIDSPath, read_raw_bids
from create_debug_report import create_debug_report
def initial_files(subject_id, main_location, function, raw_data_type):
    """
    Load MEG data and head position, create necessary folders for output.

    Args:
        subject_id (str): Identifier for the subject.
        main_location (str): Main directory where the data is stored.
        function (str): Subdirectory for function-specific data.
        raw_data_type (str): Type of data to load ('raw' or other formats like 'data').

    Returns:
        tuple: Loaded MEG data, head position data (if available), and the output folder path.
    """
    # Convert main location to a string, if not already
    main_location = str(os.getenv('MAINMEG'))
    
    # Read configuration for BIDS structure
    cfg = read_cfg(main_location)
    step = 'BIDS_structure_read'
    cfg = cfg[step]
 
    # Define necessary folder paths
    verbose_folder_path = os.path.join(main_location, "verbose")
    head_pos = None  # Initialize head position variable
    output_folder_path = os.path.join(main_location, function)

    # Load raw MEG data based on the data type ('raw' or other)
    if raw_data_type == "raw":
        # Adjust subject ID if working with raw BIDS data
        subject_id = subject_id[4:]
        raw_folder= os.path.join(main_location, "raw")
        bids_path = BIDSPath(subject=subject_id, root=raw_folder,**cfg)
        data = read_raw_bids(bids_path=bids_path, verbose=False)


        # Find MEG device type and store it 
        bids_complete_path=(bids_path.fpath)
        last_1_parts = bids_complete_path.parts[-1:]
        strings_after_last_dot = [part.split('.')[-1] for part in last_1_parts if '.' in part]

        device_mapping = {
            "fif": "MEGIN",
            "ds": "CTF",
            "pdf": "BTI"
        }
        # Check each string in the list
        for string in strings_after_last_dot:
            if string in device_mapping:
                os.environ["MEGDEVICE"] = device_mapping[string]
                print("MEG device type= ", device_mapping[string])
                create_debug_report(filename="MEGAP_report.txt")
                break
        else:  # If no match was found
            print("Could not determine the type of MEG device.")

        if device_mapping[string]=="BTI": # The BTI-related functions in MNE-BIDS have not been updated and contain bugs, particularly when loading the hs_file. To ensure reliability, load it manually.
            path=(bids_path.fpath)
            without_final_parts = path.parts[:-1]
            base_path = os.path.join(*without_final_parts)

            # Get the first folder name in the specified directory
            if os.path.exists(base_path):
                subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
                if subfolders:
                    # Append the first subfolder name to the base path
                    final_path = os.path.join(base_path, subfolders[0])
                    print("Final Path:", final_path)
                else:
                    print("No subfolders found in the specified directory.")
            else:
                print("The specified base path does not exist.")

            pdf_fname = os.path.join(final_path, 'c,rfDC')  # Full path to raw data file
            config_fname = os.path.join(final_path, 'config')  # Full path to config file
            headshape_fname = os.path.join(final_path, 'hs_file')  # Full path to head shape file (optional)

            # Check if files exist
            for file in [pdf_fname, config_fname, headshape_fname]:
                print(f"{file}: {'Exists' if os.path.exists(file) else 'Does not exist'}")

            # Load the BTI dataset
            data = mne.io.read_raw_bti(
                pdf_fname=pdf_fname,
                config_fname=config_fname,
                head_shape_fname=headshape_fname,
                preload=True,  # Set to True to load data into memory
            )

    else:
        # Load data from a .fif file if not raw data
        raw_file_name = f"{subject_id}.fif"
        data = mne.io.read_raw_fif(os.path.join(main_location, raw_data_type, raw_file_name),
                                   allow_maxshield='yes', preload=True, verbose=False)

    # Load head position data, if available
    head_pos_folder = os.path.join(main_location, "head_position")
    if os.path.exists(head_pos_folder):
        try:
            head_pos = mne.chpi.read_head_pos(os.path.join(head_pos_folder, subject_id))
        except:
            pass


    # Ensure output and verbose folders exist, create them if necessary
    for folder in [output_folder_path, verbose_folder_path]:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
                print(f"Folder '{folder}' created successfully.")
            except OSError as e:
                print(f"Error creating folder '{folder}': {e}")

    return data, head_pos, output_folder_path
