import os
import mne
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import pickle
from mne.preprocessing import compute_average_dev_head_t
import seaborn as sns
from initial_py import read_cfg
from initial_files import initial_files
import mne.transforms as transforms
import mne.viz as viz
from mne_bids import BIDSPath, read_raw_bids

def plot_head_position(subject_ids):
    # mne.viz.set_browser_backend('matplotlib')
    matplotlib.use('Agg')
    """
    Generates and saves head position plots for each subject.
    
    Args:
        subject_ids (list[str]): List of subject IDs.
    """ 
    
    # Get the main location from the environment variable
    main_location = os.getenv('MAINMEG')
    if main_location is None:
        raise ValueError("MAINMEG environment variable is not set")

    # Iterate over each subject ID
    for subject_id in subject_ids:
        # Load the data and folder path
        data, head_pos, folder_path = initial_files(subject_id, main_location, "plot_head_pos", "data")

        if data.info["hpi_results"]:
            # Invert the head transformation matrices
            original_head_dev_t = transforms.invert_transform(data.info["dev_head_t"])
            average_head_dev_t = transforms.invert_transform(compute_average_dev_head_t(data, head_pos))

            # Plot the head positions
            fig = viz.plot_head_positions(head_pos)

            # Add horizontal lines for original and average head positions
            for ax, val, val_ori in zip(fig.axes[::2], average_head_dev_t["trans"][:3, 3], original_head_dev_t["trans"][:3, 3]):
                ax.axhline(1000 * val, color="r")
                ax.axhline(1000 * val_ori, color="g")

            # Save the plot
            fig.savefig(os.path.join(folder_path, subject_id), dpi=400)
            plt.close('all')



def plot_psd(subject_ids, label):
    """
    Generates power spectral density (PSD) plots for each subject.

    Notes:
        The PSD plots are saved in subfolders named 'PSD_'.
    """
    mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')

    # Get the main directory from environment variables
    main_location = str(os.getenv('MAINMEG'))
    # Read the configuration settings for writing to BIDS
    cfg = read_cfg(main_location)
    step = 'BIDS_structure_write'
    cfg = cfg[step]
    for subject_id in subject_ids:

        # try:
            # Load the subject data using the initial_files function
            try:
                data, _, folder_path = initial_files(subject_id, main_location, f"PSD_{label}", label)
            except:
                subject_id_bids=subject_id[4:]
                ICA_folder= os.path.join(main_location, "ICA")
                bids_path = BIDSPath(subject=subject_id_bids, root=ICA_folder, processing='ICA', **cfg)
                _, _, folder_path = initial_files(subject_id, main_location, f"PSD_{label}", "data")
                data = read_raw_bids(bids_path=bids_path, verbose=False)

            # Load flat channel information and update the data's bad channels
            bad_channel_folder = os.path.join(main_location, "flat_channel")
            with open(os.path.join(bad_channel_folder, f'{subject_id}.pkl'), 'rb') as file:
                auto_flat_chs = pickle.load(file)
            data.info['bads'] = auto_flat_chs  # Mark bad channels
            
            eeg_indices = mne.pick_types(data.info, meg=False, eeg=True,ref_meg=False)
            eeg_channels = [data.ch_names[idx] for idx in eeg_indices]
            has_eeg = len(eeg_channels) > 0

            grad_indices = mne.pick_types(data.info, meg="grad", eeg=False,ref_meg=False)
            grad_channels = [data.ch_names[idx] for idx in grad_indices]
            has_grad = len(grad_channels) > 0

            if has_eeg:
                if has_grad:
                    grad_ax=1
                    mag_ax=2
                else:
                    grad_ax=0
                    mag_ax=1
            else:
                if has_grad:
                    grad_ax=0
                    mag_ax=1
                else:
                    mag_ax=0
            # Check if the folder path is valid, and generate the PSD plot
            if folder_path is not None:
                # Compute and plot the PSD, excluding flat channels
                fig = data.compute_psd().plot(exclude='bads')

                # First check if we have any flat channels
                if auto_flat_chs:

                    # Get channel types for all channels
                    ch_types = data.get_channel_types()

                    # Initialize empty lists for flat mag and grad channels
                    flat_mag_channels = []
                    flat_grad_channels = []

                    # Check each flat channel's type
                    for ch in auto_flat_chs:
                        if ch in data.ch_names:
                            idx = data.ch_names.index(ch)
                            if idx < len(ch_types):  # Ensure index is valid
                                ch_type = ch_types[idx]
                                if ch_type == 'mag':
                                    flat_mag_channels.append(ch)
                                elif ch_type == 'grad':
                                    flat_grad_channels.append(ch)

                    if len(flat_mag_channels)==1:
                        texttoplot_mag="Flat Sensor Excluded"
                    elif len(flat_mag_channels)>1:
                        texttoplot_mag = f"{len(flat_mag_channels)} Flat Sensors Excluded"

                    if len(flat_grad_channels)==1:
                        texttoplot_grad="Flat Sensor Excluded"
                    elif len(flat_grad_channels)>1:
                        texttoplot_grad = f"{len(flat_grad_channels)} Flat Sensors Excluded"

                    if len(flat_grad_channels)>0:
                        ax0 = fig.axes[grad_ax]

                        ax0.text(0.99, 0.55, texttoplot_grad, 
                                transform=ax0.transAxes,
                                horizontalalignment='right',
                                bbox=dict(facecolor='none', edgecolor='red', boxstyle='round,pad=0.2'))
                                   
                    if len(flat_mag_channels)>0:

                        ax1 = fig.axes[mag_ax]

                        ax1.text(0.99, 0.55, texttoplot_mag, 
                                transform=ax1.transAxes,
                                horizontalalignment='right',
                                bbox=dict(facecolor='none', edgecolor='red', boxstyle='round,pad=0.2'))

                # Save the figure as a .png file in the subject's folder
                fig.savefig(os.path.join(folder_path, f'{subject_id}.png'), dpi=400)
                print(f"PSD plot saved for subject: {subject_id}")
            else:
                print(f"Invalid folder path for subject ID: {subject_id}")

            # Close all matplotlib figures to free up memory
            plt.close('all')
        # except:
        #     pass

def plot_bad_channel(subject_ids):
    """
    Generate bad sensor plots for each subject.

    This function loads MEG data, identifies bad sensors, and generates visualizations of sensor locations
    as well as automated noisy sensor detection using heatmaps. The plots are saved in specified folders.
    """
    # Get the main directory from environment variables
    # mne.viz.set_browser_backend('matplotlib')
    matplotlib.use('Agg')
    main_location = str(os.getenv('MAINMEG'))
    MEG_device = str(os.getenv('MEGDEVICE'))
    find_meg_device()
    for subject_id in subject_ids:

        # Load MEG data for the subject
        data, _, folder_path = initial_files(subject_id, main_location, "plot_bad_channel", "data")

        # Load flat channel information
        flat_channel_path = os.path.join(main_location, "flat_channel")
        with open(os.path.join(flat_channel_path, f'{subject_id}.pkl'), 'rb') as file:
            flat_chs = pickle.load(file)

        # Load noisy and flat channel detection results
        bad_channel_path = os.path.join(main_location, "bad_channel")
        with open(os.path.join(bad_channel_path, f'{subject_id}.pkl'), 'rb') as file:
            auto_noisy_chs, auto_flat_chs, auto_scores = pickle.load(file)

        # Combine all bad channels and add to the MEG data's bad list
        data.info['bads'] = auto_noisy_chs + auto_flat_chs + flat_chs

        # Print bad channels for debugging
        bad_max = auto_noisy_chs + auto_flat_chs + flat_chs
        print(f"Bad channels for subject {subject_id}: {bad_max}")

        if MEG_device=="MEGIN":

            plot_bad_channel_maxwell (subject_id,data,auto_scores,bad_max,folder_path)

        else: # for other MEG systems

            plot_bad_channel_non_maxwell (subject_id,data,auto_scores,bad_max,folder_path)

def plot_bad_channel_non_maxwell (subject_id,data,auto_scores,bad_max,folder_path):

    # Extract the bad channel indexes and their corresponding scores
    ch_bad_idx = [idx for idx, ch_name in enumerate(data.info['ch_names']) if ch_name in bad_max]
    print(f"Bad channel indexes: {ch_bad_idx}")

    # Create a figure with subplots for sensor locations (both 2D and 3D)
    fig1 = plt.figure()
    fig1.suptitle(f"Bad channel detection for {subject_id}\nmag", fontsize=14)

    # First subplot: 3D sensor locations for magnetometers
    ax3d_mag = fig1.add_subplot(211, projection="3d")
    data.plot_sensors(ch_type="mag", kind="3d", axes=ax3d_mag,show=False)
    ax3d_mag.view_init(azim=70, elev=15)

    # Second subplot: 2D sensor locations for magnetometers
    ax2d_mag = fig1.add_subplot(212)
    data.plot_sensors(ch_type="mag", axes=ax2d_mag,show=False)
    location_folder_path = os.path.join(folder_path, "location")
    if not os.path.exists(location_folder_path):
        try:
            os.makedirs(location_folder_path)
            print(f"Folder '{location_folder_path}' created successfully.")
        except OSError as e:
            print(f"Error creating folder '{location_folder_path}': {e}")
    else:
        print(f"Folder '{location_folder_path}' already exists.")

    # Save the sensor location plot
    fig1.savefig(os.path.join(location_folder_path, f'{subject_id}.png'), dpi=400)
    del fig1
    plt.close('all')

def plot_bad_channel_maxwell (subject_id,data,auto_scores,bad_max,folder_path):

    # Extract the bad channel indexes and their corresponding scores
    ch_bad_idx = [idx for idx, ch_name in enumerate(data.info['ch_names']) if ch_name in bad_max]
    ch_bad_names = auto_scores['ch_names'][ch_bad_idx]
    scores = auto_scores['scores_noisy'][ch_bad_idx]
    bins = auto_scores['bins']
    bin_labels = [f'{start:.3f} – {stop:.3f}' for start, stop in bins]

    # Create a DataFrame for the noisy scores above the threshold
    data_to_plot = pd.DataFrame(data=scores, columns=pd.Index(bin_labels, name='Time (s)'), index=pd.Index(ch_bad_names, name='Channel'))
    data_to_plot = data_to_plot[data_to_plot > 7]
    print(f"Bad channel indexes: {ch_bad_idx}")
    print(f"Bad channel names: {ch_bad_names}")

    # Create a figure with subplots for sensor locations (both 2D and 3D)
    fig1 = plt.figure()
    fig1.suptitle(f"Bad channel detection for {subject_id}\nmag                                    grad", fontsize=14)

    # First subplot: 3D sensor locations for magnetometers
    ax3d_mag = fig1.add_subplot(221, projection="3d")
    data.plot_sensors(ch_type="mag", kind="3d", axes=ax3d_mag,show=False)
    ax3d_mag.view_init(azim=70, elev=15)

    # Second subplot: 2D sensor locations for magnetometers
    ax2d_mag = fig1.add_subplot(223)
    data.plot_sensors(ch_type="mag", axes=ax2d_mag,show=False)

    # Third subplot: 3D sensor locations for gradiometers
    ax3d_grad = fig1.add_subplot(222, projection="3d")
    data.plot_sensors(ch_type="grad", kind="3d", axes=ax3d_grad,show=False)
    ax3d_grad.view_init(azim=70, elev=15)

    # Fourth subplot: 2D sensor locations for gradiometers
    ax2d_grad = fig1.add_subplot(224)
    data.plot_sensors(ch_type="grad", axes=ax2d_grad,show=False)

    # Create a folder for storing the sensor location plots
    location_folder_path = os.path.join(folder_path, "location")
    if not os.path.exists(location_folder_path):
        try:
            os.makedirs(location_folder_path)
            print(f"Folder '{location_folder_path}' created successfully.")
        except OSError as e:
            print(f"Error creating folder '{location_folder_path}': {e}")
    else:
        print(f"Folder '{location_folder_path}' already exists.")

    # Save the sensor location plot
    fig1.savefig(os.path.join(location_folder_path, f'{subject_id}.png'), dpi=400)
    del fig1
    plt.close('all')
    # Create heatmap plots for "grad" and "mag" channels
    for ch_type in ["grad", "mag"]:
        ch_subset = auto_scores["ch_types"] == ch_type
        ch_names = auto_scores["ch_names"][ch_subset]
        scores = auto_scores["scores_noisy"][ch_subset]
        limits = auto_scores["limits_noisy"][ch_subset]

        # Create DataFrame for heatmap plotting
        bin_labels = [f"{start:.3f} – {stop:.3f}" for start, stop in bins]
        data_to_plot = pd.DataFrame(data=scores, columns=pd.Index(bin_labels, name="Time (s)"), index=pd.Index(ch_names, name="Channel"))

        # Plot heatmaps
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle(f"Automated noisy channel detection: {ch_type}", fontsize=16, fontweight="bold")

        # First heatmap: all scores
        sns.heatmap(data=data_to_plot, cmap="Reds", cbar_kws=dict(label="Score"), ax=ax[0])
        ax[0].set_title("All Scores", fontweight="bold")

        # Second heatmap: scores above the limit
        sns.heatmap(data=data_to_plot, vmin=np.nanmin(limits), cmap="Reds", cbar_kws=dict(label="Score"), ax=ax[1])
        ax[1].set_title("Scores > Limit", fontweight="bold")

        # Mark bad channels in red
        for i, ch_name in enumerate(ch_names):
            color = "red" if ch_name in ch_bad_names else 'none'
            ax[1].text(len(bins) + 1, i + 0.5, ch_name, va="center", ha="right", color=color)

        # Create folder for storing heatmap images
        score_folder_path = os.path.join(folder_path, f"scores_{ch_type}")
        if not os.path.exists(score_folder_path):
            try:
                os.makedirs(score_folder_path)
                print(f"Folder '{score_folder_path}' created successfully.")
            except OSError as e:
                print(f"Error creating folder '{score_folder_path}': {e}")
        else:
            print(f"Folder '{score_folder_path}' already exists.")

        # Save the heatmap plot
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(os.path.join(score_folder_path, f'{subject_id}.png'), dpi=400)

    # Close all figures to free up memory
    plt.close('all')

def plot_signal(subject_ids, label):
    matplotlib.use('Agg')
    """  
    Plot and save signal for each subject, generating figures with small and large durations.

    This function generates and saves two types of signal plots:
    - A plot of 5 channels over 150 seconds.
    - A plot of 10 channels over 1 second.

    """
    # Get the main MEG data directory from environment variables
    main_location = str(os.getenv('MAINMEG'))
    # Read the configuration settings for writing to BIDS
    cfg = read_cfg(main_location)
    step = 'BIDS_structure_write'
    cfg = cfg[step]

    if label == "plot_before_ica":
        process_laststep = "environment_noise"
    elif label == "plot_after_ica":
        process_laststep = "ICA"

    # Loop through each subject in the list
    for subject_id in subject_ids:

        try:
            data, _, folder_path = initial_files(subject_id, main_location, label, process_laststep)
        except:
            subject_id_bids=subject_id[4:]
            ICA_folder= os.path.join(main_location, "ICA")
            bids_path = BIDSPath(subject=subject_id_bids, root=ICA_folder, processing='ICA', **cfg)
            _, _, folder_path = initial_files(subject_id, main_location, f"PSD_{label}", "data")
            data = read_raw_bids(bids_path=bids_path, verbose=False)

        # Set the plotting backend to 'matplotlib' for MNE visualization
        mne.viz.set_browser_backend('matplotlib')

        # Define the subdirectories for saving the plots
        small_plot_dir = os.path.join(folder_path, "original")
        large_plot_dir = os.path.join(folder_path, "zoomed")

        # Create the directories if they don't already exist
        for plot_dir in [small_plot_dir, large_plot_dir]:
            if not os.path.exists(plot_dir):
                try:
                    os.makedirs(plot_dir)
                    print(f"Folder '{plot_dir}' created successfully.")
                except OSError as e:
                    print(f"Error creating folder '{plot_dir}': {e}")
            else:
                print(f"Folder '{plot_dir}' already exists.")

        # Generate the first plot: 5 channels over 150 seconds and save it in the 'SMALL' folder
        fig_small = mne.viz.plot_raw(data, n_channels=5, duration=150, show_first_samp=False, lowpass=120 , highpass=0.5)
        fig_small.savefig(os.path.join(small_plot_dir, f"{subject_id}.png"), dpi=400)
        print(f"unzoomed plot saved for subject: {subject_id}")

        # Generate the second plot: 10 channels over 1 second and save it in the 'LARGE' folder
        fig_large = mne.viz.plot_raw(data, n_channels=10, duration=1, show_first_samp=False, lowpass=120,highpass=0.5)
        fig_large.savefig(os.path.join(large_plot_dir, f"{subject_id}.png"), dpi=400)
        print(f"zoomed plot saved for subject: {subject_id}")

        # Close all open plots to free up memory
        plt.close('all')
