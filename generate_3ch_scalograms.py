import os
import numpy as np
import scipy.io
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from matplotlib import cm
import matplotlib.pyplot as plt
import getpass
import socket

# ==== CONFIGURATION ====
def get_data_path():
    username = getpass.getuser().lower()
    if username == "administrator":
        return r"C:\Users\Administrator\PycharmProjects\OSR_Project\datasets\myData\myData.mat"
    elif username == "user":
        return r"C:\Users\user\PycharmProjects\PyTorch_Project\datasets\PreambleData\myData.mat"
    hostname = socket.gethostname().lower()
    if "admin" in hostname:
        return r"C:\Users\Administrator\PycharmProjects\OSR_Project\datasets\myData\myData.mat"
    elif "user" in hostname:
        return r"C:\Users\user\PycharmProjects\PyTorch_Project\datasets\PreambleData\myData.mat"
    return r"C:\Users\Administrator\PycharmProjects\OSR_Project\datasets\myData\myData.mat"

DATA_PATH = get_data_path()
BASE_OUTPUT_DIR = os.path.abspath("track_A_wavelet_scalograms_png_channel3")  # <-- changed
IMG_SIZE = 128
SPLIT_RATIO = 0.8

TX_RANGES = {
    "Tx01": (0, 350), "Tx02": (351, 701), "Tx03": (774, 1123), "Tx04": (1234, 1583),
    "Tx05": (1623, 1972), "Tx06": (2025, 2374), "Tx07": (2478, 2827), "Tx08": (2916, 3265),
    "Tx09": (3366, 3715), "Tx10": (3761, 4110), "Tx11": (4225, 4574),
}

WAVELETS = [
    ("pywt_Morlet", "morl", "pywt", None),
    ("pywt_Mexh", "mexh", "pywt", None),
    ("pywt_Gaus8", "gaus8", "pywt", None),
    ("ssq_GMW", "gmw", "ssqueezepy.cwt", None),
    ("ssq_Bump", "bump", "ssqueezepy.cwt", None),
    ("ssq_CMHat", "cmhat", "ssqueezepy.cwt", None),
    ("ssq_SST_GMW", "gmw", "ssqueezepy.ssq_cwt", {"wavelet_kwargs": {"mu": 5, "sigma": 1}}),
    ("ssq_SST_Bump", "bump", "ssqueezepy.ssq_cwt", {}),
    ("ssq_SST_HHat", "hhhat", "ssqueezepy.ssq_cwt", {}),
]

IMPAIRMENT_SWEEPS = {
    "AWGN": [0, 5, 10, 15, 20, 25, 30],
    "Rician": [0, 5, 10, 15, 30],
    "Doppler": [0, 50, 100, 200, 400, 800],
}
BASELINES = {"AWGN": 0, "Rician": 0, "Doppler": 0}

def get_cwt():
    try:
        from ssqueezepy.cwt import cwt
    except ImportError:
        try:
            from ssqueezepy import cwt
        except ImportError:
            raise ImportError("Neither 'from ssqueezepy.cwt import cwt' nor 'from ssqueezepy import cwt' worked. Please check your ssqueezepy installation.")
    return cwt

def apply_awgn(iq_data, snr_db):
    if snr_db == 0:
        noise = np.random.normal(0, np.std(iq_data), size=iq_data.shape)
        return iq_data + noise
    signal_power = np.mean(np.abs(iq_data)**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=iq_data.shape)
    return iq_data + noise

def apply_rician(iq_data, K_dB):
    K = 10 ** (K_dB / 10) if K_dB != 0 else 0.0
    N = len(iq_data)
    LOS = np.sqrt(K / (K + 1)) if K > 0 else 0
    NLOS = np.sqrt(1 / (2 * (K + 1))) * (np.random.normal(0, 1, N) + 1j * np.random.normal(0, 1, N)) if K > 0 else np.random.normal(0, 1, N)
    h = LOS + NLOS
    return iq_data * h

def apply_doppler(iq_data, fd, fs=20e6):
    if fd == 0:
        return iq_data
    N = len(iq_data)
    t = np.arange(N) / fs
    doppler_shift = np.exp(1j * 2 * np.pi * fd * t)
    iq_data_c = iq_data.astype(np.complex128) if not np.iscomplexobj(iq_data) else iq_data
    return iq_data_c * doppler_shift

def compute_scalogram(signal, wavelet_code, package, params=None, img_size=IMG_SIZE):
    if package == "pywt":
        import pywt
        scales = np.geomspace(1, len(signal), num=64)
        coeffs, _ = pywt.cwt(signal, scales, wavelet_code)
        scalogram = np.abs(coeffs)
    elif package == "ssqueezepy.cwt":
        cwt = get_cwt()
        Wx, *_ = cwt(signal, wavelet=wavelet_code)
        scalogram = np.abs(Wx)
    elif package == "ssqueezepy.ssq_cwt":
        import ssqueezepy
        params = params or {}
        wavelet_kwargs = params.get('wavelet_kwargs', None)
        supports_kw = 'wavelet_kwargs' in ssqueezepy.ssq_cwt.__code__.co_varnames
        if wavelet_kwargs and supports_kw:
            out = ssqueezepy.ssq_cwt(signal, wavelet=wavelet_code, wavelet_kwargs=wavelet_kwargs)
        else:
            out = ssqueezepy.ssq_cwt(signal, wavelet=wavelet_code)
        ssq = out[1] if isinstance(out, tuple) and len(out) > 1 else out[0]
        scalogram = np.abs(ssq)
    else:
        raise ValueError(f"Unknown package: {package}")
    if scalogram.shape != (img_size, img_size):
        scalogram = resize(scalogram, (img_size, img_size), order=1, mode='reflect', anti_aliasing=True)
    scalogram = np.log1p(scalogram)
    scalogram = (scalogram - np.min(scalogram)) / (np.max(scalogram) - np.min(scalogram) + 1e-8)
    return scalogram.astype(np.float32)

def scalogram_to_3ch(scalogram, method="colormap"):
    if method == "colormap":
        cmap = cm.get_cmap('viridis')
        rgb_img = cmap(scalogram)[:, :, :3]
        return rgb_img.astype(np.float32)
    elif method == "diffs":
        first_deriv = np.gradient(scalogram)[0]
        second_deriv = np.gradient(first_deriv)[0]
        stacked = np.stack([scalogram, first_deriv, second_deriv], axis=-1)
        stacked = (stacked - stacked.min(axis=(0,1))) / (stacked.max(axis=(0,1)) - stacked.min(axis=(0,1)) + 1e-8)
        return stacked.astype(np.float32)
    else:
        raise ValueError(f"Unknown 3ch conversion method: {method}")

def save_png(img_arr, out_path):
    # img_arr is (H,W,3) float32 in [0,1]
    plt.imsave(out_path, img_arr, format='png')

def main():
    print("Working directory:", os.getcwd())
    print("Scalograms will be saved to:", BASE_OUTPUT_DIR)
    print("Loading .mat data from:", DATA_PATH)
    data = scipy.io.loadmat(DATA_PATH)
    orig_ltf_data = np.abs(data['myData']['LTF'][0, 0])

    # --- One-at-a-time ablation: 18 experiments ---
    for impairment, sweep in IMPAIRMENT_SWEEPS.items():
        for value in sweep:
            awgn_val = value if impairment == "AWGN" else BASELINES["AWGN"]
            rician_val = value if impairment == "Rician" else BASELINES["Rician"]
            doppler_val = value if impairment == "Doppler" else BASELINES["Doppler"]
            experiment_dir = f"{impairment}_{value}"
            print(f"\nExperiment: {experiment_dir} (AWGN={awgn_val}, Rician={rician_val}, Doppler={doppler_val})")
            for wavelet_dir, wavelet_code, package, params in WAVELETS:
                print(f" \n Processing wavelet: {wavelet_dir}")
                for tx, (start, end) in TX_RANGES.items():
                    tx_signals = orig_ltf_data[start:end, :]
                    n_signals = tx_signals.shape[0]
                    indices = np.arange(n_signals)
                    train_idx, test_idx = train_test_split(
                        indices, train_size=SPLIT_RATIO, test_size=1-SPLIT_RATIO,
                        random_state=42, shuffle=True
                    )
                    splits = {'train': train_idx, 'test': test_idx}
                    for split, split_indices in splits.items():
                        split_dir = os.path.join(BASE_OUTPUT_DIR, experiment_dir, wavelet_dir, split, tx)
                        os.makedirs(split_dir, exist_ok=True)
                        for idx, sig_idx in enumerate(tqdm(split_indices, desc=f"{wavelet_dir} {split} {tx} ({experiment_dir})")):
                            signal = tx_signals[sig_idx]
                            impaired_signal = apply_awgn(signal, awgn_val)
                            impaired_signal = apply_rician(impaired_signal, rician_val)
                            impaired_signal = apply_doppler(impaired_signal, doppler_val)
                            try:
                                scal_in = np.real(impaired_signal) if np.iscomplexobj(impaired_signal) else impaired_signal
                                scalogram = compute_scalogram(scal_in, wavelet_code, package, params)
                                scalogram_3ch = scalogram_to_3ch(scalogram, method="colormap")
                                # Save as PNG
                                out_path = os.path.join(split_dir, f"sample_{sig_idx+1:05d}.png")
                                save_png(scalogram_3ch, out_path)
                            except Exception as e:
                                print(f"Error processing {wavelet_dir} {split} {tx} signal {sig_idx+1} ({experiment_dir}): {e}")

    # --- Combined-average run: 1 experiment ---
    avg_awgn = int(round(np.mean(IMPAIRMENT_SWEEPS["AWGN"])))
    avg_rician = int(round(np.mean(IMPAIRMENT_SWEEPS["Rician"])))
    avg_doppler = int(round(np.mean(IMPAIRMENT_SWEEPS["Doppler"])))
    print(f"\nExperiment: Combined_Average (AWGN={avg_awgn}, Rician={avg_rician}, Doppler={avg_doppler})")
    for wavelet_dir, wavelet_code, package, params in WAVELETS:
        print(f" \n Processing wavelet: {wavelet_dir}")
        for tx, (start, end) in TX_RANGES.items():
            tx_signals = orig_ltf_data[start:end, :]
            n_signals = tx_signals.shape[0]
            indices = np.arange(n_signals)
            train_idx, test_idx = train_test_split(
                indices, train_size=SPLIT_RATIO, test_size=1-SPLIT_RATIO,
                random_state=42, shuffle=True
            )
            splits = {'train': train_idx, 'test': test_idx}
            for split, split_indices in splits.items():
                split_dir = os.path.join(BASE_OUTPUT_DIR, "Combined_Average", wavelet_dir, split, tx)
                os.makedirs(split_dir, exist_ok=True)
                for idx, sig_idx in enumerate(tqdm(split_indices, desc=f"{wavelet_dir} {split} {tx} (Combined_Average)")):
                    signal = tx_signals[sig_idx]
                    impaired_signal = apply_awgn(signal, avg_awgn)
                    impaired_signal = apply_rician(impaired_signal, avg_rician)
                    impaired_signal = apply_doppler(impaired_signal, avg_doppler)
                    try:
                        scal_in = np.real(impaired_signal) if np.iscomplexobj(impaired_signal) else impaired_signal
                        scalogram = compute_scalogram(scal_in, wavelet_code, package, params)
                        scalogram_3ch = scalogram_to_3ch(scalogram, method="colormap")
                        # Save as PNG
                        out_path = os.path.join(split_dir, f"sample_{sig_idx+1:05d}.png")
                        save_png(scalogram_3ch, out_path)
                    except Exception as e:
                        print(f"Error processing {wavelet_dir} {split} {tx} signal {sig_idx+1} (Combined_Average): {e}")

    print("\nAll 19 experiments per wavelet (18 ablations + 1 combined-average) completed for all wavelets.")
    print("3-channel PNG scalograms are in:", BASE_OUTPUT_DIR)

if __name__ == "__main__":
    main()