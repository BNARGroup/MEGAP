# Import necessary libraries
import json
import sys
import os
import shutil
import datetime
from contextlib import contextmanager
from mne.utils import logger
import re

@contextmanager
def redirect_stdout(log_file, step_name):
    """
    Context manager to temporarily redirect stdout to a log file for logging purposes.

    Args:
        log_file (file object): The file where stdout will be redirected.
        step_name (str): The name of the current step to log.
        
    """
    # Save the original stdout
    original_stdout = sys.stdout
    # Redirect stdout to the provided file
    sys.stdout = log_file

    try:
        # Log the step name and the current timestamp at the beginning
        current_time = datetime.datetime.now()
        print("____________________" + step_name + "____________________")
        print("Start time:", current_time.strftime("%Y-%m-%d %H:%M:%S"), "\n")
        yield log_file
    finally:
        # Restore the original stdout after logging is done
        sys.stdout = original_stdout


def read_cfg(config_path):
    """
    Read and parse a configuration file in JSON format.

    Args:
        config_path (str): Path to the main directory containing the config folder.

    Returns:
        dict: Parsed configuration data as a dictionary.
    """
    # Construct the full path to the configuration file
    cfg_file_path = os.path.join(config_path, "config", "Pipeline_config.cfg")
    
    # Open and load the JSON configuration file
    with open(cfg_file_path) as json_file:
        config_data = json.load(json_file)
    
    # Check for 'None' values in the JSON data and convert them to Python None
    config_data = replace_none_values(config_data)
    
    return config_data


def replace_none_values(data):
    """
    Recursively replace string 'None' values with Python's None in a nested dictionary.

    Args:
        data (dict): The dictionary to check and modify.

    Returns:
        dict: The modified dictionary with 'None' values replaced by None.
    """
    clean_data = data.copy()  # Copy the data to avoid modifying the original

    # Iterate through the dictionary
    for key, value in data.items():
        if isinstance(value, dict):
            # Recursively check nested dictionaries
            clean_data[key] = replace_none_values(value)
        elif value == 'None':
            # Replace string 'None' with Python's None
            clean_data[key] = None
    
    return clean_data

def write_warning(subject_id,warning_message):
            
    main_location = str(os.getenv('MAINMEG'))
    # Define necessary folder paths
    warning_folder_path = os.path.join(main_location, "warning")

    if not os.path.exists(warning_folder_path):
        try:
            os.makedirs(warning_folder_path)
            print("Warning folder created successfully.")
        except OSError as e:
            print(f"Error creating warning folder : {e}")

    # Append the warning to the file
    with open(os.path.join(warning_folder_path, f'{subject_id}.txt'), "a") as warning_file:
        warning_file.write(warning_message)

def find_meg_device():
    file_path="MEGAP_Report.txt"
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Match a line starting with 'meg_device = '
                match = re.match(r"Meg Device:\s*(.+)", line)
                if match:
                    # Extract and return the value
                    os.environ["MEGDEVICE"] = match.group(1).strip()
    except Exception as e:
        print(f"Error reading file: {e}")

