# Import necessary libraries
import mne
from ica_labeler import ica_labeler
import matplotlib.pyplot as plt
import os 
import numpy as np
from mne.preprocessing import ICA
from mne.preprocessing import (
    create_eog_epochs,
    create_ecg_epochs,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg
from initial_files import initial_files
from initial_py import read_cfg, redirect_stdout, write_warning
import matplotlib
import os
from scipy.ndimage import zoom
import numpy as np
import pandas as pd
from kneed import KneeLocator
import time
from mne_bids import BIDSPath, write_raw_bids

def ica_decomposition(subject_ids):
    matplotlib.use('Agg')
    current_time = time.time()
    """
    Apply Independent Component Analysis (ICA) decomposition to MEG data and save the cleaned data.

    """
    # mne.viz.set_browser_backend('matplotlib')
    main_location = str(os.getenv('MAINMEG'))
    cfg = read_cfg(main_location)
    step = 'ica_decomposition'
    cfg = cfg[step]
    cfg=cfg['ecg']

    cfg_apply = read_cfg(main_location)
    step_apply = 'ica_decomposition'
    cfg_apply=cfg_apply[step_apply]['apply']

    for subject_id in subject_ids:
        # Create a log file for the subject
        log_file_path = os.path.join(main_location, 'verbose', f'{subject_id}.txt')
        with open(log_file_path, 'a') as log_file:
            with redirect_stdout(log_file, step):
                print(f"Config= {cfg}\n")

                # Load raw data and prepare folder
                
                raw, _, folder_path = initial_files(subject_id, main_location, "ICA", "environment_noise")

                # Filter the raw data and downsample for the CNN model.
                filt_raw = raw.copy().filter(l_freq=1.0, h_freq=100)
                filt_raw.resample(sfreq=250)

                
                folder_path_images = os.path.join(main_location, "plot_ICA", subject_id)

                # Create a folder for saving ICA images if it doesn't exist
                if not os.path.exists(folder_path_images):
                    try:
                        os.makedirs(folder_path_images)
                        print(f"Folder '{folder_path_images}' created successfully.")
                    except OSError as e:
                        print(f"Error creating folder '{folder_path_images}': {e}")
                else:
                    print(f"Folder '{folder_path_images}' already exists.")

                # Plot and save EOG and ECG evoked responses
                try:
                    eog_evoked = create_eog_epochs(filt_raw).average()
                    eog_evoked.apply_baseline(baseline=(None, -0.2))
                    fig_list = eog_evoked.plot_joint()
                    i=0
                    try:
                        for fig in fig_list:
                            fig.savefig(os.path.join(folder_path_images, f"{subject_id}-{i}"), dpi=400)
                            i += 1
                    except: 
                        fig_list.savefig(os.path.join(folder_path_images, f"{subject_id}-{i}"), dpi=400)
                        i += 1
                    del fig_list
                except:
                    print("without eog_evoked")

                ecg_evoked = create_ecg_epochs(filt_raw).average()
                ecg_evoked.apply_baseline(baseline=(None, -0.2))
                fig_list = ecg_evoked.plot_joint()

                i=0
                try:
                    for fig in fig_list:
                        fig.savefig(os.path.join(folder_path_images, f"{subject_id}-{i}"), dpi=400)
                        i += 1
                except: 
                    fig_list.savefig(os.path.join(folder_path_images, f"{subject_id}-{i}"), dpi=400)
                    i += 1
                del fig_list

                # proc_info = raw.info.get('proc_history', [])[0]
                # cov_elbow = proc_info['max_info']['sss_info']['nfree']
                
                meg_ch_idx = mne.pick_types(raw.info, meg=True , ref_meg=False)
                data_pca = raw.get_data(picks=meg_ch_idx)
                pca = mne.utils._PCA(n_components=None, whiten=True)
                data = pca.fit_transform(data_pca.T)
                use_ev = pca.explained_variance_ratio_
                
                x = range(1, len(use_ev)+1)
                elbow_point_index = KneeLocator(
                    x, use_ev, curve='convex', direction='decreasing')
                print("elbow rank",elbow_point_index.knee)
                elbow_rank = int(elbow_point_index.knee)

                if elbow_rank < 15:
                    check_elbowpoint(subject_id, elbow_rank)
                    elbow_rank=30

                # Apply ICA decomposition
                ica = ICA(n_components=elbow_rank, max_iter="auto", random_state=97, method='picard')
                ica.fit(filt_raw)

                # Print explained variance ratio
                try: # Just for MEGIN
                    explained_var_ratio = ica.get_explained_variance_ratio(filt_raw, ch_type='grad')
                    for channel_type, ratio in explained_var_ratio.items():
                        print(f"Fraction of {channel_type} variance explained by all components (grad): {ratio}")
                except: 
                    pass

                explained_var_ratio = ica.get_explained_variance_ratio(filt_raw, ch_type='mag')
                for channel_type, ratio in explained_var_ratio.items():
                    print(f"Fraction of {channel_type} variance explained by all components (mag): {ratio}")

                # Plot and save ICA components
                filt_raw.load_data()
                fig_list = ica.plot_components()

                try:
                    for fig in fig_list:
                        fig.savefig(os.path.join(folder_path_images, f"{subject_id}-{i}"), dpi=400)
                        i += 1
                except: 
                    fig_list.savefig(os.path.join(folder_path_images, f"{subject_id}-{i}"), dpi=400)
                    i += 1
                del fig_list
                # Exclude bad components detected from EOG and ECG patterns
                ica.exclude = []

                ica_topoplott=ica_topoplot(ica)
                ica_components = ica.get_sources(filt_raw) 
                ica_components = ica_components.get_data()
                ica_labels=ica_labeler(ica_components,ica_topoplott)


                exclude = np.where(ica_labels != 0)
                ecg_indices, ecg_scores = ica.find_bads_ecg(
                    filt_raw, **cfg)

                ica.exclude = ecg_indices

                dctLabelConv = {'Brain': 0, 'n': 0, 0: 0,
                                'Blink': 1, 'b': 1, 1: 1,
                                'Cardiac': 2, 'c': 2, 2: 2,
                                'Saccad': 3, 's': 3, 3: 3,
                                }

                indices_with_labels = [(i, list(dctLabelConv.keys())[list(dctLabelConv.values()).index(
                    label)]) for i, label in enumerate(ica_labels) if dctLabelConv.get(label) in {1, 2, 3}]
                print(indices_with_labels)

                # Plot and save ICA scores
                try:
                    fig = ica.plot_scores(ecg_scores, title='ecg')
                    fig.savefig(os.path.join(folder_path_images, f"{subject_id}-{i}"), dpi=400)
                    i += 1
                    del fig
                except:
                    print("without ecg_scores")


                ica.exclude=[]
                ica.exclude =list(set(list(exclude[0]) + ecg_indices))
                print("ica.exclude= ",ica.exclude)
                
                try:
                # Plot diagnostics and save
                    fig_list = ica.plot_properties(filt_raw, picks=ica.exclude)
                    for fig in fig_list:
                        fig.savefig(os.path.join(folder_path_images, f"{subject_id}-{i}"), dpi=400)
                        i += 1
                    del fig_list
                except:
                    print("without plot_properties")
                    

                try:
                    # Plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
                    fig = ica.plot_sources(eog_evoked, title='eog')
                    fig.savefig(os.path.join(folder_path_images, f"{subject_id}-{i}"), dpi=400)
                    i += 1
                    del fig
                except:
                    print("without plot_sources eog")


                try:
                    # Plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
                    fig = ica.plot_sources(ecg_evoked, title='ecg')
                    fig.savefig(os.path.join(folder_path_images, f"{subject_id}-{i}"), dpi=400)
                    i += 1
                    del fig
                except:
                    print("without plot_sources ecg")

                try:
                    # Plot all components
                    if ica.n_components_<16:
                        fig = ica.plot_sources(filt_raw, picks=range(0, ica.n_components_), show_scrollbars=False, start=0, stop=10, title='Temporal ICA')
                        fig.savefig(os.path.join(folder_path_images, f"{subject_id}-{i}"), dpi=400)
                        i += 1
                        del fig
                    else:
                        for j in range(ica.n_components_ // 15):
                            d = j * 15
                            fig = ica.plot_sources(filt_raw, picks=range(d, d + 15), show_scrollbars=False, start=0, stop=10, title='Temporal ICA')
                            fig.savefig(os.path.join(folder_path_images, f"{subject_id}-{i}"), dpi=400)
                            i += 1
                            del fig

                        fig = ica.plot_sources(filt_raw, picks=range(d+15, ica.n_components_), show_scrollbars=False, start=0, stop=10, title='Temporal ICA')
                        fig.savefig(os.path.join(folder_path_images, f"{subject_id}-{i}"), dpi=400)
                        i += 1
                        del fig
                except:
                    print("without plot_sources ecg")
                # Apply ICA to the raw data
                ica.apply(raw, exclude=ica.exclude,**cfg_apply)

                # Save the cleaned data
                write_bids(subject_id,raw,folder_path) # In BIDS structure
                # raw.save(os.path.join(folder_path, f"{subject_id}.fif"), overwrite=True, fmt='double') # Without using BIDS structure
                try:
                    del filt_raw,raw,data
                except:
                    print("ram cleaning not compeleted")
                plt.close('all') 

                print("elbow rank= ",elbow_rank)
                print(current_time-time.time())

        check_ica_rejection(subject_id,ica)
 

def ica_topoplot(ica):
    matplotlib.use('Agg')
    topoplott = np.empty((ica.n_components, 120, 120, 3))
    for i in range(0, ica.n_components):
        p = ica.plot_components(picks=i, show_names=False, outlines=None,
                                contours=15, res=100, title=None, sensors=False, image_interp='linear')
        canvas = FigureCanvasAgg(p)
        canvas.draw()

        # Get the RGB buffer from the canvas
        width, height = p.get_size_inches() * p.get_dpi()
        rgb_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(
            int(height), int(width), 4)

        rgb_array = rgb_array[80:-30, 20:-20, :]
        target_size = (120, 120, 3)

        # Calculate the scale factor for downsampling
        scale_factor = (
        target_size[0] / rgb_array.shape[0],
        target_size[1] / rgb_array.shape[1],
        1  # No scaling along the third dimension (channels)
        )

    # Use the zoom function to downsample the image
        rgb_array = zoom(rgb_array, scale_factor, order=1)

        # Convert RGBA to RGB (discard the alpha channel)
        rgb_array = rgb_array[:, :, :3]
        plt.imshow(rgb_array)
        plt.axis('off')  #  Turn off axis labels and ticks
        plt.close()
        topoplott[i, :, :, :] = rgb_array
    return topoplott



def write_bids(subject_id,raw,folder_path):
    """
    Convert MEG data into BIDS format and write it to disk.

    The function reads preprocessed MEG data for each subject and writes the data in BIDS format.
    """
    main_location = str(os.getenv('MAINMEG'))
    # Read the configuration settings for writing to BIDS
    cfg = read_cfg(main_location)
    step = 'BIDS_structure_write'
    cfg = cfg[step]

    # Set cHPI channels to 'misc' type (if any exist)
    chpi_channels = {ch_name: 'misc' for ch_name in raw.ch_names if ch_name.startswith('CHPI')}
    if chpi_channels:
        raw.set_channel_types(chpi_channels)  

    # Modify the subject ID to match BIDS format 
    subject_id_bids = subject_id[4:]

    # Define the BIDS path using the BIDSPath function and configuration
    bids_path = BIDSPath(subject=subject_id_bids, root=folder_path, processing='ICA', **cfg)

    # # Write the raw data in BIDS format
    write_raw_bids(raw, bids_path=bids_path, verbose=False, allow_preload=True, format="FIF", overwrite=True)
    # print(f"Data written to BIDS for subject: {subject_id}")


def check_ica_rejection(subject_id,ica):

    main_location = str(os.getenv('MAINMEG'))
    cfg = read_cfg(main_location)
    cfg_warning = cfg['warning']
    ica_threshold=cfg_warning['ica']

    # Get the list or array of indices of rejected components
    rejected_components = ica.exclude
    warning_message = "\n"
    # Check if the number of rejected components is one or none
    if len(rejected_components) <= 1:
        warning_message += (
            "\n" + "_" * 10 + "ICA" + "_" * 10 + "\n" +
            f"{len(rejected_components)} ICA component(s) have been rejected. Please verify if this is correct."
        )
        write_warning(subject_id, warning_message)

    
    elif len(rejected_components) >= ica_threshold:
        warning_message += ("\n"+"_"*10+"ICA"+"_"*10+"\n"+
            f"{len(rejected_components)} ICA components have been rejected, "
            "which may indicate a large number of artifacts in the data. Please review the rejection process."
        )
        write_warning(subject_id, warning_message)

def check_elbowpoint(subject_id, elbow):
    # Get the list or array of indices of rejected components
    warning_message = "\n"
    # Check if the number of rejected components is less than 15
    if elbow < 15:
        warning_message += (
            "\n" + "_" * 10 + "ICA" + "_" * 10 + "\n" +
            f"The elbow point  is {elbow}, which is insufficient for reliable ICA decomposition. "
            "The value of 15 will be used instead. Please verify the data."
        )
        write_warning(subject_id, warning_message)
