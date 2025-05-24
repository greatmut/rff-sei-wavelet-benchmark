import os
import sys
import glob
import subprocess

# ========== CONFIGURATION ==========
# Update to PNG root!
SCALOGRAM_ROOT = "track_A_wavelet_scalograms_png_channel3"
MAT_FILE_PATHS = [
    r"C:\Users\Administrator\PycharmProjects\OSR_Project\datasets\myData\myData.mat",
    r"C:\Users\user\PycharmProjects\PyTorch_Project\datasets\PreambleData\myData.mat"
]
SCALOGRAM_GENERATOR_SCRIPT = "generate_3ch_scalograms.py"
TRAIN_EVAL_SCRIPT = "cnn_train_eval_report.py"
WAVELET_LIST = [
    "pywt_Morlet", "pywt_Mexh", "pywt_Gaus8",
    "ssq_GMW", "ssq_Bump", "ssq_CMHat",
    "ssq_SST_GMW", "ssq_SST_Bump", "ssq_SST_HHat"
]
EXPERIMENT_LIST = [
    "AWGN_0", "AWGN_5", "AWGN_10", "AWGN_15", "AWGN_20", "AWGN_25", "AWGN_30",
    "Rician_0", "Rician_5", "Rician_10", "Rician_15", "Rician_30",
    "Doppler_0", "Doppler_50", "Doppler_100", "Doppler_200", "Doppler_400", "Doppler_800",
    "Combined_Average"
]
TX_LIST = [f"Tx{i:02d}" for i in range(1, 12)]  # Tx01 ... Tx11

def check_scalogram_files(root_dir, wavelets, experiments, tx_list):
    """
    Checks if all required PNG scalogram folders exist, and each contains at least one .png file.
    Returns True if all present, False otherwise.
    """
    missing = []
    for exp in experiments:
        for wave in wavelets:
            for split in ['train', 'test']:
                for tx in tx_list:
                    png_dir = os.path.join(root_dir, exp, wave, split, tx)
                    if not os.path.isdir(png_dir) or not glob.glob(os.path.join(png_dir, "*.png")):
                        missing.append(png_dir)
    if missing:
        print(f"Missing {len(missing)} PNG scalogram folders or files.")
        for m in missing[:10]:  # Show a few missing folders
            print("  MISSING or EMPTY:", m)
        if len(missing) > 10:
            print(f"  ...and {len(missing) - 10} more.")
        return False
    print("All PNG scalogram folders with images are present.")
    return True

def choose_mat_path():
    for path in MAT_FILE_PATHS:
        if os.path.isfile(path):
            print("Using MAT file:", path)
            return path
    print("ERROR: No .mat file found in expected locations!")
    sys.exit(1)

def run_scalogram_generation(mat_path):
    print("Running scalogram generation...")
    # You can call as a subprocess, or import & call a function if refactored.
    result = subprocess.run([sys.executable, SCALOGRAM_GENERATOR_SCRIPT], check=True)
    print("Scalogram generation completed.")

def run_training_evaluation():
    print("Running model training/evaluation...")
    result = subprocess.run([sys.executable, TRAIN_EVAL_SCRIPT], check=True)
    print("Model training and evaluation completed.")

def main():
    print("=== OSR Automated Pipeline ===")
    # Step 1: Check if scalogram files are present
    data_ready = check_scalogram_files(SCALOGRAM_ROOT, WAVELET_LIST, EXPERIMENT_LIST, TX_LIST)

    # Step 2: If not present, generate scalograms
    if not data_ready:
        mat_path = choose_mat_path()
        run_scalogram_generation(mat_path)
        # Recheck
        data_ready = check_scalogram_files(SCALOGRAM_ROOT, WAVELET_LIST, EXPERIMENT_LIST, TX_LIST)
        if not data_ready:
            print("ERROR: Scalogram generation failed or incomplete.")
            sys.exit(2)

    # Step 3: Run training and evaluation
    run_training_evaluation()

    print("=== All tasks completed. ===")

if __name__ == "__main__":
    main()