![logo_dark](https://github.com/user-attachments/assets/391b2709-4349-496b-8420-40f80741fcc3)

# MEGAP: MEG Automatic Pipeline

Magnetoencephalography (MEG) data often contains noise and artifacts that necessitate comprehensive pre-processing prior to analysis. **MEGAP** is developed to automate this crucial pre-processing phase, minimizing the requirement for manual adjustments. By utilizing standardized parameter selections, MEGAP facilitates the pre-processing of large-scale MEG datasets, enhancing the consistency, efficiency, and reproducibility of research outcomes.

## Project Purpose

The primary aim of MEGAP is to optimize the pre-processing of MEG data, specifically for large-scale datasets that are impractical to handle manually. Researchers dealing with extensive MEG datasets will find the automation capabilities of MEGAP advantageous, as it conserves time and mitigates the risk of human error during the pre-processing stage.

## Documentation
The MEGAP [documentation](https://megap.gitbook.io/megap) includes comprehensive details such as installation guidelines, folder structure, usage instructions, and in-depth explanations of each step in the workflow.

## Features of MEGAP
- **Automated Execution**: The pipeline runs to completion without requiring user intervention.
- **Standardization**: Parameters are aligned with guidelines and recommendations from various studies.
- **Configuration Files**: Users can adjust parameters for specific datasets as needed.
- **Organized Output**: Output files are systematically arranged into folders for each participant, conforming to the Brain Imaging Data Structure (BIDS) standard for MEG (MEG-BIDS).
- **Intermediate Saving Points**: MEGAP can (re)start from any intermediate point by saving the outputs of each stage.
- **Step-Wise Results Visualization**: Results from each processing step are plotted to verify outputs.
- **Verbose Files**: Detailed output logs of each function are saved in text files for each participant.
- **Warning System**:This system generates a warning text, which flags potential issues based on user-defined thresholds.

### How to Run MEGAP

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/BNARGroup/MEGAP.git
   cd MEGAP
   ```

2. **Install Required Packages**:  
   Install necessary packages, including MNE, MNE-BIDS, TensorFlow, MATLAB Engine, and any other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Copy Required Repositories**:  
   Clone the following repositories into the config folder of MEGAP:
   - [mne-matlab](https://github.com/mne-tools/mne-matlab)
   - [MegNET_2020](https://github.com/DeepLearningForPrecisionHealthLab/MegNET_2020)
   - [zapline-plus](https://github.com/MariusKlug/zapline-plus)
   - [eeglab](https://github.com/sccn/eeglab)

4. **Add Data to MEGAP**:  
   Place MEG data organized in the MEG-BIDS structure into the `result/raw` folder of MEGAP. Additionally, include the fine calibration file and a cross-talk compensation file in the `result/config` folder.

5. **Execute the MEGAP Script**:  
   You can now run MEGAP using the `MEGAP.py` script and let it pre-process all of your data without any manual intervention.

   MEGAP will automatically verify all necessary folders and notify you if anything is missing.
   
   ```bash
   +--------------------------------------------+--------+
   |               Required Item                | Status |
   +--------------------------------------------+--------+
   |               Config Folder                | ✔️ OK  |
   |                 Raw Folder                 | ✔️ OK  |
   |           CFG Configuration File           | ✔️ OK  |
   |       SSS Calibration File (MEGIN only)    | ✔️ OK  |
   |          Cross-talk File (MEGIN only)      | ✔️ OK  |
   | Zapline-plus-main folder from their github | ✔️ OK  |
   |  eeglab-develop folder from their github   | ✔️ OK  |
   | mne-matlab-master folder from their github | ✔️ OK  |
   |  CNN Model folder from MegNET_2020 github  | ✔️ OK  |
   +--------------------------------------------+--------+
   *** All necessary folders exist for running MEGAP ***
   ```

6. **Outputs**:  
   Upon completion of the pipeline, the output power spectra plots can be located in the ‘PSD’ folders, while the output text files will be found in the ‘verbose’ folder. The supplementary materials provide a detailed description of the folder structure and contents in the ‘Details of MEGAP Folder Structure’ section.

## Citation
If you find our code or paper helpful in your research, we kindly ask you to cite our [publication](google.com).

## License

This project is licensed under the MIT License. You are free to modify and distribute it in accordance with the license terms. For more information, refer to the [LICENSE](LICENSE) file.
