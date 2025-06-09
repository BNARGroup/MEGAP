# Import necessary libraries
import os
import mne
from initial_files import initial_files
from initial_py import read_cfg, redirect_stdout
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from initial_py import find_meg_device

def environment_noise(subject_ids):
    """
    Apply Maxwell filtering to MEG data and save the filtered data for each subject.

    This function reads configuration files, applies Maxwell filtering to the MEG data to remove
    noise and interference, and saves the filtered data in FIF format.
    """
    # Get the main directory location from an environment variable
    main_location = str(os.getenv('MAINMEG'))
    MEG_device = str(os.getenv('MEGDEVICE'))
    find_meg_device()
    
    # Read the configuration settings for the 'filter_maxwell' step
    cfg = read_cfg(main_location)
    step = 'filter_maxwell'
    cfg = cfg[step]
 
    cfg_switch = read_cfg(main_location)["apply_environment_denoising"]["switch"]

    # Loop over each subject
    for subject_id in subject_ids:
        # Create a log file to store verbose output for each subject
        log_file_path = os.path.join(main_location, 'verbose', f'{subject_id}.txt')
        with open(log_file_path, 'a') as log_file:
            with redirect_stdout(log_file, step):
                print(f"Processing subject: {subject_id}")
                if MEG_device=="MEGIN":
                    print(f"Configuration used for Maxwell filtering: {cfg}\n")

                    # Load the MEG data and initialize folder structure
                    data, head_pos, folder_path = initial_files(subject_id, main_location, "environment_noise", "OTP")
                    if cfg_switch==True:
                        # Apply Maxwell filtering to the raw MEG data
                        data_sss = mne.preprocessing.maxwell_filter(
                            data,
                            **cfg,
                            calibration=os.path.join(main_location, "config", "sss_cal.dat"),  # Path to calibration file
                            cross_talk=os.path.join(main_location, "config", "ct_sparse.fif"),  # Path to cross-talk file
                            head_pos=head_pos,
                            skip_by_annotation=('BAD_muscle')  # Skip bad muscle
                        )
                    else:
                        data_sss=data
                    # Interpolate bad channels using minimum-norm method
                    data_sss.interpolate_bads(reset_bads=False)
                else:

                    data_sss, head_pos, folder_path = initial_files(subject_id, main_location, "environment_noise","multi_taper_removal") #we should change the name "filter_maxwell" to "environment denoising"
                                   
                    flat_channel_path = os.path.join(main_location, "flat_channel")
                    with open(os.path.join(flat_channel_path, f'{subject_id}.pkl'), 'rb') as file:
                        flat_channels = pickle.load(file)

                    bad_channel_path = os.path.join(main_location, "bad_channel")
                    with open(os.path.join(bad_channel_path, f'{subject_id}.pkl'), 'rb') as file:
                        auto_noisy_chs, auto_flat_chs, _ = pickle.load(file)

                    # Combine all bad channels into the data's bad channel list
                    data_sss.info['bads'] = auto_noisy_chs + auto_flat_chs + flat_channels
                    # Interpolate bad channels using minimum-norm method
                    data_sss.interpolate_bads(reset_bads=False) # interpolating bad sensors before using 3GC or regression (ref burkhard)
                    
                    if cfg_switch==True:
                        if MEG_device=="CTF":
                            data_sss.apply_gradient_compensation(3) # Gradient compensation will be applied at the end of pipeline
                        else:
                            data_sss=BTI_denoising(data_sss) # Maybe you need to downgrade your MNE version


                # Load muscle artifact annotations (Again for CTF and BTI)
                muscle_annotation_path = os.path.join(main_location, "annotate_muscle")
                with open(os.path.join(muscle_annotation_path, f'{subject_id}.pkl'), 'rb') as file:
                    muscle_annotations = pickle.load(file)

                # Set the muscle artifact annotations in the data
                data_sss.set_annotations(muscle_annotations)
                

                # Save the filtered data in FIF format
                filtered_data_path = os.path.join(folder_path, f"{subject_id}.fif")
                data_sss.save(filtered_data_path, overwrite=True, fmt='double')
                print(f"Filtered Maxwell data saved for subject {subject_id} at {filtered_data_path}\n")


def BTI_denoising (data,decim_fit=100):
    
    meg_picks = mne.pick_types(data.info, meg=True, ref_meg=False)
    ref_picks = mne.pick_types(data.info, meg=False, ref_meg=True)

    # Extract reference and MEG data
    ref_data = data.get_data(picks=ref_picks)
    meg_data = data.get_data(picks=meg_picks)

    decim_fit = 100

    estimator = Pipeline([('scaler', StandardScaler()), ('estimator', LinearRegression())])

    # Prepare data for fitting
    X = ref_data[:, ::decim_fit].T
    Y = meg_data[:, ::decim_fit].T

    # Fit and predict
    Y_pred = estimator.fit(X, Y).predict(ref_data.T)

    # Subtract predicted values
    data._data[meg_picks] -= Y_pred.T
    return data
