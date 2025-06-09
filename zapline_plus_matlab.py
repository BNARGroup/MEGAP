# Import necessary libraries
import os
import matlab.engine
import mne
import numpy as np
import datetime
from initial_files import initial_files
from initial_py import read_cfg, redirect_stdout, write_warning,find_meg_device


def zapline_plus_matlab(subject_ids):
    """
    Apply the ZapLine_plus using a MATLAB script to remove powerline noise from MEG data.

    This function interacts with a MATLAB engine to run the ZapLine_plus.
    """
    # Get the main location of the MEG data from environment variables
    main_location = str(os.getenv('MAINMEG'))
    MEG_device = str(os.getenv('MEGDEVICE'))
    find_meg_device()
    step = 'zapline'
     
    # Loop through each subject to process their MEG data
    for subject_id in subject_ids:
        # Create a log file to capture the verbose output for the current subject
        log_file_path = os.path.join(main_location, 'verbose', f'{subject_id}.txt')
        grad_channel_indices=[]
        mag_channel_indices=[]
        with open(log_file_path, 'a') as log_file:
            with redirect_stdout(log_file, step):
                # Load the MEG data
                if MEG_device=="MEGIN":
                    data, _, _ = initial_files(subject_id, main_location, 'zapline_plus', 'filter_chpi')
                
                    # Extract powerline noise frequency and sampling rate from the MEG data
                    powerline_noise_freq = data.info['line_freq']   # Powerline noise frequency

                    # Extract cHPI frequencies and channel indices for magnetometers and gradiometers
                    chpi_frequencies, _, _ = mne.chpi.get_chpi_info(data.info)
                    grad_channel_indices = mne.pick_types(data.info, meg="grad",ref_meg=False)
                    mag_channel_indices = mne.pick_types(data.info, meg="mag",ref_meg=False)
                else:
                    data, _, _ = initial_files(subject_id, main_location, 'zapline_plus', 'data')
                    powerline_noise_freq=[]
                    chpi_frequencies=[]
                    mag_channel_indices = mne.pick_types(data.info, meg=True , ref_meg=True)

                sampling_rate = float(int(data.info['sfreq'])   )          # Sampling rate

                # Start the MATLAB engine and run the ZapLine-plus MATLAB function
                eng = matlab.engine.start_matlab()
                zapline_output=eng.zapline_matlab(main_location, subject_id, grad_channel_indices, mag_channel_indices, chpi_frequencies, powerline_noise_freq, sampling_rate)
                output_array = np.array(zapline_output)

                # Access Nrem_mag and Nrem_grad
                Nrem_mag = output_array[0][0]
                Nrem_grad = output_array[0][1]


                check_zapline_plus(subject_id,Nrem_mag,Nrem_grad)
                print(f"ZapLine processing completed for subject {subject_id}.")

                current_time = datetime.datetime.now()
                print("Completation time:", current_time.strftime("%Y-%m-%d %H:%M:%S"), "\n")



def check_zapline_plus(subject_id,Nrem_mag,Nrem_grad):

    main_location = str(os.getenv('MAINMEG'))
    cfg_warning = read_cfg(main_location)['warning']
    zapline_threshold=cfg_warning['zapline_plus']

    # Check if Nrem_mag or Nrem_grad exceeds the threshold
    warning_message = "\n"

    if (Nrem_mag) > zapline_threshold:
        warning_message += ("\n"+"_"*10+"Zapline_Plus"+"_"*10+"\n"+
                            f"Warning: zapline_plus removed {(Nrem_mag)} magnetometer components, "
                            f"which exceeds the threshold of {zapline_threshold} components.\n"
                            f"This suggests the presence of significant line noise in the data. "
                            "Please check the magnetometer data.\n")

    if (Nrem_grad) > zapline_threshold:
        warning_message += ("\n"+"_"*10+"Zapline_Plus"+"_"*10+"\n"+
                            f"Warning: zapline_plus removed {(Nrem_grad)} gradiometer components, "
                            f"which exceeds the threshold of {zapline_threshold} components.\n"
                            f"This suggests the presence of significant line noise in the data. "
                            "Please check the gradiometer data .\n")

    # If any of the conditions are met, write the warning
    if warning_message:
        write_warning(subject_id, warning_message)
    else:
        print(f"No issues with line noise removal for {subject_id} in zapline plus.")