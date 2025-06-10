import os
import shutil
from IPython.display import clear_output
from  prettytable import PrettyTable

def print_megap():
    print("""
  ███╗   ███╗ ███████╗  ██████╗   █████╗  ██████╗ 
  ████╗ ████║ ██╔════╝ ██╔════╝  ██╔══██╗ ██╔══██╗
  ██╔████╔██║ █████╗   ██║  ███╗ ███████║ ██████╔╝ 
  ██║╚██╔╝██║ ██╔══╝   ██║   ██║ ██╔══██║ ██╔═══╝
  ██║ ╚═╝ ██║ ███████╗ ╚██████╔╝ ██║  ██║ ██║ 
  ╚═╝     ╚═╝ ╚══════╝  ╚═════╝  ╚═╝  ╚═╝ ╚═╝
--------------MEG Automatic Pipeline--------------    """)

    print("""References: 
   Please cite our publication if MEGAP is used.
        """)


def copy_file(source_folder, destination_folder, file_name):
    # Construct full file paths
    source_file = os.path.join(source_folder, file_name)
    destination_file = os.path.join(destination_folder, file_name)
    
    # Check if the source file exists
    if not os.path.exists(source_file):
        print(f"Source file '{file_name}' not found in '{source_folder}'")
        return
    
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Copy the file
    shutil.copy(source_file, destination_file)
    print(f"File '{file_name}' copied from '{source_folder}' to '{destination_folder}'")



def list_files():
    """
    List of subject files from the 'raw' folder of the MEG data directory.

    The function checks if the necessary folders and files exist and lists all files starting with 'sub' in the 'raw' folder.

    Returns:
        list: A list of subject files matching the criteria.
    """
    # Get the main MEG directory from environment variables
    main_folder_path = str(os.getenv("MAINMEG"))
    verbose_folder_path = os.path.join(main_folder_path, "verbose")

    # Check if the necessary configuration files and folders exist
    check_config_folder()

    # Copy edited clean_data_with_zapline_plus.m to main repository
    source_folder = os.path.dirname(main_folder_path)
    destination_folder = os.path.join(main_folder_path,"config" , "zapline-plus-main")
    file_name = 'clean_data_with_zapline_plus.m'
    copy_file(source_folder, destination_folder, file_name)

    # Ensure the 'verbose' folder exists, create it if necessary
    if not os.path.exists(verbose_folder_path):
        try:
            os.makedirs(verbose_folder_path)
            print(f"Folder '{verbose_folder_path}' created successfully.")
        except OSError as e:
            print(f"Error creating folder '{verbose_folder_path}': {e}")
    else:
        pass

    try:
        # Define the path to the 'raw' folder
        raw_folder_path = os.path.join(main_folder_path, "raw")

        # Get a list of all files and directories in the 'raw' folder
        all_files = os.listdir(raw_folder_path)

        # Filter out files that start with 'sub' (assumed subject files)
        subject_files = [f for f in all_files if f.startswith('sub')]

        # Print the list of subject files
 
        if len(subject_files)>1:
            print("subject IDs: " , subject_files)
            print(f"{len(subject_files)} subjects found.")
        else:
            print("subject ID: " , subject_files)
            print(f"{len(subject_files)} subject found.")

        return subject_files

    except OSError as e:
        # Handle error if the 'raw' folder cannot be accessed
        print(f"Error reading files in {raw_folder_path}: {e}")
        return []



def check_config_folder():
    """
    Check for the existence of essential configuration files and folders in the MEG directory.

    This function checks for the following:
    - Config folder
    - Raw data folder
    - CFG configuration file
    - SSS calibration file
    - Cross-talk file
    - CNN model file
    """
    # Get the main MEG directory from environment variables
    main_folder_path = str(os.getenv("MAINMEG"))
    config_folder_path = os.path.join(main_folder_path, "config")

    # Paths to required folders and files
    required_paths = {
        "Config Folder": config_folder_path,
        "Raw Folder": os.path.join(main_folder_path, "raw"),
        "CFG Configuration File": os.path.join(config_folder_path, "Pipeline_config.cfg"),
        "SSS Calibration File (MEGIN only)": os.path.join(config_folder_path, "sss_cal.dat"),
        "Cross-talk File (MEGIN only)": os.path.join(config_folder_path, "ct_sparse.fif"),
        "Zapline-plus-main folder from their github": os.path.join(config_folder_path, "zapline-plus-main"),
        "eeglab-develop folder from their github": os.path.join(config_folder_path, "eeglab-develop"),
        "mne-matlab-master folder from their github": os.path.join(config_folder_path, "mne-matlab-master"),
        "CNN Model folder from MegNET_2020 github": os.path.join(config_folder_path, "MegNET_2020-main","MEGnet","model","MEGnet_final_model.h5")
    }

    # Clear console and print MEGAP
    clear_output(wait=False)
    print_megap()


    # Create a table with PrettyTable
    table = PrettyTable()
    table.field_names = ["Required Item", "Status"]

    # Check if each required path exists
    for name, path in required_paths.items():
        if os.path.exists(path):
            table.add_row([name, "✔️ OK"])  # Add a check mark if exists
        else:
            table.add_row([name, "❌ Missing"])  # Mark as missing if not found

    # Print the table
    print(table)


    # Check if each required path exists
    for name, path in required_paths.items():
        if os.path.exists(path):
            pass
        else:
            raise ValueError(f"{name} does not exist!") from None
    print("*** All necessary folders exist for running MEGAP ***")
    print("\n")
