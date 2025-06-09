# Import necessary libraries
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.utils import logger
from mne.preprocessing import annotate_muscle_zscore
from initial_files import initial_files
from initial_py import read_cfg, redirect_stdout, write_warning

def muscle_artifact(subject_ids):
    """
    Detects and annotates muscle artifacts in MEG data for a list of subjects.

    This function reads configuration files, logs information about the process, and saves
    annotated muscle artifacts for each subject in a pickle (.pkl) file.
    """
    
    # Get main location of MEG data from environment variable
    main_location = str(os.getenv('MAINMEG'))
    
    # Read configuration settings for the 'muscle' step
    cfg = read_cfg(main_location)
    step = 'annotate_muscle'
    cfg = cfg[step]

    cfg_warning = read_cfg(main_location)['warning']
    limit_percentage=cfg_warning['muscle']

    # Loop over each subject
    for subject_id in subject_ids:
        # Create a log file for the current subject to log processing steps
        log_file_path = os.path.join(main_location, 'verbose', f'{subject_id}.txt')
        with open(log_file_path, 'a') as log_file:
            with redirect_stdout(log_file, step):
                print(f"Processing subject: {subject_id}")
                print(f"Configuration used: {cfg}\n")

                # Load raw MEG data and create appropriate folder paths for outputs
                data, _, folder_path = initial_files(subject_id, main_location, "annotate_muscle", "multi_taper_removal")
                _, _, folder_path_plot = initial_files(subject_id, main_location, "plot_muscle_zscore", "multi_taper_removal")
                
                # Annotate muscle artifacts using z-score method
                annot_muscle, scores_muscle = annotate_muscle_zscore(
                    data,
                    **cfg  # Pass configuration parameters for muscle annotation
                )

                total_duration = data.times[-1]  # Duration of the raw data in seconds
                total_annotation_duration = sum(annot_muscle.duration)

                # Check if the annotations' duration exceeds 10% of the total duration
                if total_annotation_duration > (limit_percentage / 100) * total_duration:
                    warning_message = ("\n"+"_"*10+"Muscle Artifact"+"_"*10+"\n"+
                    f"muscle annotations covering {total_annotation_duration:.2f}s, more than {limit_percentage}% of total duration ({total_duration:.2f}s). "
                    f"Check the data and Z-score parameter (current value: {cfg['threshold']}).\n")
                    write_warning(subject_id,warning_message)
                else:
                    print(f"Annotations are within acceptable limits for {subject_id}.")
                                        
                # Save the annotations and scores into a pickle file for further analysis
                output_path = os.path.join(folder_path, f'{subject_id}.pkl')
                with open(output_path, 'wb') as file:
                    pickle.dump(annot_muscle, file)

                fig, ax = plt.subplots()
                ax.plot(data.times, scores_muscle)
                ax.axhline(y=cfg["threshold"], color="r")
                ax.set(xlabel="time, (s)", ylabel="zscore", title="Muscle activity")

                fig.savefig(os.path.join(folder_path_plot, subject_id), dpi=400)
                plt.close()
                del fig

                print(f"Annotations for subject {subject_id} saved to {output_path}\n")



