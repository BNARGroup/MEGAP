from plotting import plot_head_position, plot_bad_channel, plot_psd, plot_signal
from ica_decomposition import ica_decomposition
from head_position import head_position
from annotate import muscle_artifact
from filter_chpi import filter_chpi, chpi_crop
from line_noise import multi_taper_removal
from bad_channel import bad_channel_detection, flat_channel
from environment_noise import environment_noise
from zapline_plus_matlab import zapline_plus_matlab
from OTP import OTP
from list_files import list_files
import os
import warnings

 
warnings.filterwarnings("ignore")
os.environ["MAINMEG"] = os.path.join(os.getcwd(), "result")
os.environ["MEGDEVICE"] = "MEGIN"
subject_ids = list_files()

for subject_id in subject_ids:

    subject_id = [subject_id]  

    # 1. Crop the data segments containing extraneous data for each subject ID.
    # chpi_crop(subject_id)

    # # 2. Detect and mark flat channels in the MEG data for each subject ID.
    # flat_channel(subject_id)

    # # 3. Plot the power spectral density (PSD) of the MEG data for each subject ID, with the label "data".
    # plot_psd(subject_id, label="data")

    # # 4. Estimate the head position for each subject ID.
    # head_position(subject_id)

    # # 5. Plot the estimated head position for each subject ID.
    # plot_head_position(subject_id)

    # # 6. Filter the continuous head position indicator (cHPI) signals in the MEG data for each subject ID.
    # filter_chpi(subject_id)

    # # 7. Plot the power spectral density (PSD) of the filtered cHPI data for each subject ID, with the label "filter_chpi".
    # plot_psd(subject_id, label="filter_chpi")

    # # 8. Use MATLAB's ZapLine_plus algorithm to remove line noise artifacts (e.g., 50/60Hz powerline noise) from the MEG data for each subject ID.
    # zapline_plus_matlab(subject_id)

    # # 9. Plot the PSD of the ZapLine-plus cleaned  data for each subject ID, with the label "zapline_plus".
    # plot_psd(subject_id, label="zapline_plus")

    # # 10. Remove any remaining line noises using multi-taper removal (regression-based method).
    # multi_taper_removal(subject_id)

    # # 11. Plot the PSD of the MEG data after multi-taper line noise removal, with the label "multi_taper_removal".
    # plot_psd(subject_id, label="multi_taper_removal")

    # # 12. Annotate muscle artifacts in the MEG data for each subject ID.
    # muscle_artifact(subject_id)

    # # 13. Identify and mark bad channels in the MEG data for each subject ID.
    # bad_channel_detection(subject_id)

    # # 14. Plot the identified bad channels for each subject ID.
    # plot_bad_channel(subject_id)

    # # 15. Apply Oversampled Temporal Projection (OTP) to the MEG data for each subject ID.
    # OTP(subject_id)

    # # 16. Plot the PSD of the MEG data after applying OTP for each subject ID, with the label "OTP".
    # plot_psd(subject_id, label="OTP")

    # # 17. Apply Maxwell filtering to the MEG data for each subject ID.
    # environment_noise(subject_id)

    # # # 18. Plot the PSD of the MEG data after Maxwell filtering for each subject ID, with the label "filter_maxwell".
    # plot_psd(subject_id, label="environment_noise")

    # # # # # # 19. Plot the raw MEG signal before performing Independent Component Analysis (ICA) for each subject ID, with the label "signal_before_ica".
    # plot_signal(subject_id, label="plot_before_ica")

    # # 20. Perform ICA decomposition on the MEG data for each subject ID to identify and remove artifacts.
    # ica_decomposition(subject_id) 

    # 21. Plot the PSD of the MEG data after ICA decomposition for each subject ID, with the label "ICA".
    plot_psd(subject_id, label="ICA")

    # 22. Plot the raw MEG signal after ICA decomposition for each subject ID, with the label "signal_after_ica".
    plot_signal(subject_id, label="plot_after_ica")
