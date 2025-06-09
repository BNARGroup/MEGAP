# Copyright (c) 2021 The University of Texas Southwestern Medical Center.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted for academic and research use only (subject to the limitations in the disclaimer below) provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of the copyright holders nor the names of its
#       contributors may be used to endorse or promote products derived from this
#       software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# For more information, please check https://github.com/DeepLearningForPrecisionHealthLab/MegNET_2020

# Import necessary libraries
import os
import numpy as np
# from tensorflow import keras
import keras
import tensorflow_addons as tfa  


def calculate_start_times(input_length, window_length=15000, overlap_length=3750):
    """
    Calculate start times for windowing a time series with overlap.

    Args:
        input_length (int): Total length of the input time series.
        window_length (int): Length of each window (default is 15000, representing 60 seconds at 250 Hz).
        overlap_length (int): Length of overlap between windows (default is 3750, representing 15 seconds at 250 Hz).

    Returns:
        list: A list of start times for each window in the time series.
    """
    start_times = []
    current_start = 0
    # Loop to calculate start times until the input length is exhausted
    while current_start + window_length <= input_length:
        start_times.append(current_start)
        current_start += window_length - overlap_length
    return start_times


def predict_and_vote(model, time_series_list, spatial_maps_list, ground_truth_list, window_length=15000, overlap_length=3750):
    """
    Predict and vote for each window of the input time series data.

    Args:
        model (keras.Model): The pre-trained model for prediction.
        time_series_list (list): List of time series arrays.
        spatial_maps_list (list): List of spatial maps corresponding to the time series.
        ground_truth_list (list): Ground truth labels for each time series (can use dummy values if not available).
        window_length (int): Length of each window for prediction (default is 15000).
        overlap_length (int): Overlap between windows (default is 3750).

    Returns:
        tuple: Predictions, ground truth votes, chunk-wise predictions, chunk-wise ground truth.
    """
    # Lists to store predictions and ground truths
    prediction_votes = []
    ground_truth_votes = []
    chunk_predictions = []
    chunk_ground_truths = []

    # Loop over each time series and its corresponding spatial map
    for time_series, spatial_map, ground_truth in zip(time_series_list, spatial_maps_list, ground_truth_list):
        time_series_length = time_series.shape[0]

        # Get start times for each window using the overlap
        start_times = calculate_start_times(time_series_length, window_length, overlap_length)

        # If the last window isn't fully covered, add the last valid window
        if start_times[-1] + window_length <= time_series_length:
            start_times.append(time_series_length - window_length)

        # Initialize a dictionary to track voting weights for each time window
        time_window_weights = {start: 0 for start in start_times}

        # Distribute votes over each time window
        for t in range(time_series_length):
            matching_windows = [start <= t < start + window_length for start in time_window_weights.keys()]
            windows_containing_t = np.sum(matching_windows)
            for start_time, is_in_window in zip(time_window_weights.keys(), matching_windows):
                if is_in_window:
                    time_window_weights[start_time] += 1.0 / windows_containing_t

        # Dictionary to store weighted predictions
        weighted_predictions = {}
        
        # Make predictions for each time window and accumulate votes
        for start_time in time_window_weights.keys():
            prediction = model.predict([np.expand_dims(spatial_map, 0),
                                        np.expand_dims(np.expand_dims(time_series[start_time:start_time + window_length], 0), -1)],verbose=False)
            chunk_predictions.append(prediction)
            chunk_ground_truths.append(ground_truth)
            weighted_predictions[start_time] = prediction * time_window_weights[start_time]

        # Combine weighted predictions to get the final vote for each window
        combined_predictions = np.stack([weighted_predictions[key] for key in sorted(weighted_predictions.keys())]).mean(axis=0)
        combined_predictions /= combined_predictions.sum()
        
        # Append the final predictions and ground truths
        prediction_votes.append(combined_predictions)
        ground_truth_votes.append(ground_truth)

    return np.stack(prediction_votes), np.stack(ground_truth_votes), np.stack(chunk_predictions), np.stack(chunk_ground_truths)


def ica_labeler(time_series_array, spatial_map_array):
    """
    Perform ICA component labeling using pre-trained MEGnet model.

    Args:
        time_series_array (np.ndarray): Array containing time series data of shape [N, T] where N is the number of samples and T is the time length.
        spatial_map_array (np.ndarray): Array containing spatial maps of shape [N, 120, 120, 3].

    Returns:
        np.ndarray: Predicted labels for ICA components.
    """
    main_location = str(os.getenv('MAINMEG'))
    model_directory = os.path.join(main_location, "config","MegNET_2020-main")

    # Validate input shapes
    try:
        assert time_series_array.shape[0] == spatial_map_array.shape[0], "Time series and spatial maps must have the same number of samples."
        assert spatial_map_array.shape[1:] == (120, 120, 3), "Spatial maps must have shape [N, 120, 120, 3]."
        assert time_series_array.shape[1] >= 15000, "Time series must be at least 60 seconds long (60*250=15000)."
    except AssertionError as e:
        raise ValueError(str(e))

    # Load the pre-trained model
    model = keras.models.load_model(os.path.join(model_directory,'MEGnet','model', 'MEGnet_final_model.h5')) 

    # Perform prediction and voting on the input data
    predictions_vote, ground_truth_vote, predictions_chunk, ground_truth_chunk = predict_and_vote(
        model, time_series_array, spatial_map_array, np.zeros((spatial_map_array.shape[0], 3)), window_length=15000, overlap_length=3750)

    # Decide the format of the output
    output_type = 'list'  # Change this if output should be in array format
    output_directory = None  # Optionally set the output directory to save predictions

    # Format predictions based on the selected output type
    if output_type.lower() == 'array':
        final_output = predictions_vote[:, 0, :]
    else:
        final_output = predictions_vote[:, 0, :].argmax(axis=1)

    # Optionally save the predictions to a file
    if output_directory is not None:
        output_path = os.path.join(output_directory, 'ICA_component_labels.txt')
        np.savetxt(output_path, final_output)

    return final_output
