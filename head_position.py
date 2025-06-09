# Import necessary libraries
import os
import pickle
import mne
import datetime
from initial_files import initial_files
from initial_py import read_cfg, redirect_stdout, write_warning
from mne.preprocessing import annotate_movement
from initial_py import find_meg_device
def head_position(subject_ids):
    """
    Calculate and save head position based on cHPI (continuous Head Position Indicator) data for each subject.

    This function computes head positions, cHPI amplitudes, and cHPI locations, and saves the results.
    """
    # Get the main directory location from an environment variable
    main_location = str(os.getenv('MAINMEG'))
    MEG_device = str(os.getenv('MEGDEVICE'))
    find_meg_device()

    # Read the configuration for the 'head_position' step
    step = 'head_position'
    cfg = read_cfg(main_location)[step]
    cfg_warning = read_cfg(main_location)['warning']
    limit_movement=cfg_warning['movement']

    # Loop over each subject
    for subject_id in subject_ids:
        # Create a log file to store verbose output for each subject
        log_file_path = os.path.join(main_location, 'verbose', f'{subject_id}.txt')
        with open(log_file_path, 'a') as log_file:
            with redirect_stdout(log_file, step):
                print(f"Processing subject: {subject_id}")
                # Load the MEG data, head position data, and folder path
                data, head_pos, folder_path = initial_files(subject_id, main_location, 'head_position', 'data')

                # Load previously detected flat channels
                folder_path_flat_channel = os.path.join(main_location, "flat_channel")
                with open(os.path.join(folder_path_flat_channel, f'{subject_id}.pkl'), 'rb') as file:
                    auto_flat_chs = pickle.load(file)
                data.info['bads'] = auto_flat_chs  # Mark flat channels as bad

                if data.info["hpi_results"]:
                    if MEG_device=="MEGIN":
                        print(f"Configuration used for head position calculation: {cfg}\n")
                        # Extract cHPI coil information
                        _, ch_idx, chpi_codes = mne.chpi.get_chpi_info(data.info)
                        print(f'cHPI coil indices for {subject_id}: {ch_idx}')
                        print(f'cHPI coil codes for {subject_id}: {chpi_codes}\n')

                        # Compute cHPI amplitudes using the configuration provided
                        chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(data, **cfg['amplitudes'])
                        print("cHPI amplitude calculation keys:", chpi_amplitudes.keys())

                        # Compute cHPI locations
                        chpi_locs = mne.chpi.compute_chpi_locs(data.info, chpi_amplitudes, **cfg['locs'])
                        print("cHPI location calculation keys:", chpi_locs.keys())
                    else:   
                        chpi_locs = mne.chpi.extract_chpi_locs_ctf(data)

                    # Compute head position from cHPI locations
                    head_pos = mne.chpi.compute_head_pos(data.info, chpi_locs, **cfg['head_pos'])

                    # Save the computed head position data to a file
                    head_pos_file_path = os.path.join(folder_path, subject_id)
                    mne.chpi.write_head_pos(head_pos_file_path, head_pos)
                    print(f"Head position saved at: {head_pos_file_path}\n")

                    head_movement_warning(subject_id,data,head_pos,limit_movement)
                else:
                    print("Could not find any cHPI data")

                
                current_time = datetime.datetime.now()
                print("Completation time:", current_time.strftime("%Y-%m-%d %H:%M:%S"), "\n")


def head_movement_warning(subject_id,data,head_pos,limit_movement):
                
    annotation_movement, _ = annotate_movement(data, head_pos, mean_distance_limit=limit_movement)
    total_duration = data.times[-1]  # Duration of the raw data in seconds
    time_exceeding_threshold  = sum(annotation_movement.duration)

    # Calculate percentage of data exceeding the threshold
    percent_exceeding_threshold = (time_exceeding_threshold / total_duration) * 100
    percent_exceeding_threshold_rounded = round(percent_exceeding_threshold, 2)

    if percent_exceeding_threshold > 0:  # If there is any time exceeding the threshold
            warning_message = ("\n"+"_"*10+"Head Movement"+"_"*10+"\n"+
                f"Head movement annotations detected exceeding the limit of {limit_movement * 1000:.2f}mm.\n"
                f"{time_exceeding_threshold:.2f}s of the dataset exceed this threshold,"
                f"which is {percent_exceeding_threshold_rounded:.2f}% of the total dataset duration.\n"
                "Check the head movements plot." )
            
            write_warning(subject_id,warning_message)