function zapline = zapline_matlab(main_folder, subject_id, grad_index, mag_index, chpi_freqs, powerline_noise, sampling_rate)
    % Function to apply ZapLine filtering on MEG data to remove powerline noise.
    % Args:
    %   main_folder (str): Path to the main folder containing subject data.
    %   subject_id (str): Subject ID.
    %   grad_index (array): Indices of gradiometer channels.
    %   mag_index (array): Indices of magnetometer channels.
    %   chpi_freqs (array): cHPI frequencies.
    %   powerline_noise (float): Powerline noise frequency (e.g., 50 or 60 Hz).
    %   sampling_rate (float): Sampling rate of the MEG data.
    % Returns:
    %   zapline (int): Number of components which zapline_plus rejected

    % Define the paths for logging and dependencies
    log_file_path = fullfile(main_folder, "verbose", [subject_id + ".txt"]);
    eeglab_path = fullfile(main_folder, "config", "eeglab-develop");
    mne_matlab_path = fullfile(main_folder, "config", "mne-matlab-master");
    zapline_path = fullfile(main_folder, "config", "zapline-plus-main");
    
    % Add paths for required toolboxes and dependencies
    addpath(genpath(eeglab_path));
    addpath(genpath(mne_matlab_path));
    addpath(genpath(zapline_path));
    
    % Start EEGLAB
    eeglab;
    
    % Define input and output folder paths for data
    if isempty(chpi_freqs)
        input_folder = fullfile(main_folder, "data");
    else
        input_folder = fullfile(main_folder, "filter_chpi");
    end

    output_folder = fullfile(main_folder, "zapline_plus");


    
    % Define input and output file names
    subject_fif_file = subject_id + ".fif";
    infile = fullfile(input_folder, subject_fif_file);
    outfile = fullfile(output_folder, subject_fif_file);
    Nrem_mag=0
    Nrem_grad=0
    try
        % Calculate harmonics
        harmonics = powerline_noise * (1:floor(min(chpi_freqs) / powerline_noise));
        
        % Find the maximum harmonic
        max_harmonic = max(harmonics) + 1;

        
        % Display the results (optional)
        disp('Harmonics calculated successfully.');
        disp(['Harmonics: ', num2str(harmonics)]);
        disp(['Max harmonic: ', num2str(max_harmonic)]);
    catch ME
        max_harmonic = 260; % Assign a default value
        disp('Setting max_harmonic to default value: 260');
    end

    

    % Load constants for MNE processing if not already loaded
    global FIFF;
    if isempty(FIFF)
        FIFF = fiff_define_constants();
    end
    
    % Attempt to load raw MEG data
    try
        raw = fiff_setup_read_raw(infile);
    catch
        error('MNE:mne_ex_read_write_raw', '%s', mne_omit_first_line(lasterr));
    end
    
    % Create output folder for processed data
    mkdir(output_folder);
    [outfid, cals] = fiff_start_writing_raw(outfile, raw.info);
    
    % Define sample range for processing
    first_sample = raw.first_samp;
    last_sample = raw.last_samp;
    sample_range = last_sample - first_sample + 1;
    
    % Initialize flag for first data buffer
    first_buffer = true;
    
    % Process data in chunks
    for first = first_sample:sample_range:last_sample
        last = first + sample_range - 1;
        if last > last_sample
            last = last_sample;
        end
        
        % Read raw data segment
        try
            [data, ~] = fiff_read_raw_segment(raw, first, last);
        catch
            fclose(raw.fid);
            fclose(outfid);
            error('MNE:mne_ex_read_write_raw', '%s', mne_omit_first_line(lasterr));
        end

        % Check if magnetometer indices are not empty
        if ~isempty(mag_index)
            % Apply ZapLine filtering for magnetometer channels
            [cleaned_mag_data, ~, result_mag, ~] = clean_data_with_zapline_plus(data(mag_index + 1, :), sampling_rate, [4, max_harmonic]);
            data(mag_index + 1, :) = cleaned_mag_data;
            fprintf("Magnetometers processing DONE\n");

            % Logging results into the verbose file
            diary(log_file_path);
            freq_mag = round(mean(result_mag.noisePeaks, 2), 1);
            disp("Magnetometer line noise frequencies:");
            disp(freq_mag');
            
            Nrem_mag = round(mean(result_mag.NremoveFinal, 2), 1);
            disp("Number of components removed from magnetometers:");
            disp(Nrem_mag');
            diary off;
        else
            fprintf("Magnetometer indices are empty. Skipping processing.\n");
        end

        % Check if gradiometer indices are not empty
        if ~isempty(grad_index)
            % Apply ZapLine filtering for gradiometer channels
            [cleaned_grad_data, ~, result_grad, ~] = clean_data_with_zapline_plus(data(grad_index + 1, :), sampling_rate, [4, max_harmonic]);
            data(grad_index + 1, :) = cleaned_grad_data;
            fprintf("Gradiometers processing DONE\n");
            
            % Logging results into the verbose file
            diary(log_file_path);
            freq_grad = round(mean(result_grad.noisePeaks, 2), 1);
            disp("Gradiometer line noise frequencies:");
            disp(freq_grad');
            
            Nrem_grad = round(mean(result_grad.NremoveFinal, 2), 1);
            disp("Number of components removed from gradiometers:");
            disp(Nrem_grad');
            diary off;
        else
            fprintf("Gradiometer indices are empty. Skipping processing.\n");
        end

        zapline = [sum(Nrem_mag), sum(Nrem_grad)]

        % Handle the first buffer
        if first_buffer
            if first > 0
                fiff_write_int(outfid, FIFF.FIFF_FIRST_SAMPLE, first);
            end
            first_buffer = false;
        end
        
        % Write the cleaned buffer to the output file
        fiff_write_raw_buffer(outfid, data, cals);
    end

    % Finish writing the processed raw data file
    fiff_finish_writing_raw(outfid);

    % Close all open figures
    close all;
end
