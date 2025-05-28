import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, periodogram, hilbert
from scipy.stats import kurtosis
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import joblib

# Noise Filters
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=1.0, highcut=200.0, fs=1000.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data, axis=0)

# Apply after bandpass
def notch_filter(data, freq=60.0, fs=1000.0, quality=30.0):
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, data, axis=0)

# Noise Metrics for evaluation
def compute_rmse(true, estimate):
    return np.sqrt(np.mean((true - estimate) ** 2))

# Kurtosis signal reduction > 0 shows a denoised signal
def proportion_of_positive_kurtosis_signals(kurtosis_raw, kurtosis_denoised):
    return (np.array([(kurtosis_raw - kurtosis_denoised) > 0]).sum() / len(kurtosis_raw)) * 100

# Use a Standard scaler to reduce the mean to 0 and std to 1

# Computing the power envelope of each channel

def band_power_envelope(ecog_signal: np.ndarray, lowcut: float, highcut: float, fs: float = 1000.0, order: int = 4) -> np.ndarray:
    """Computes band-limited envelope via Hilbert transform.
    Parameters
    ----------
    self.ecog_signal : np.ndarray (T, channels)
        This is the ecog signal that has been filtered.
    lowcut : float
        This is the lower band limit in Hz.
    highcut : float
        This is the upper band limit in Hz.
    fs : float, optional
        This is the frequency of the sample., by default 1000.0
    order : int, optional
        This is the Butterworth order, by default 4
    Returns
    -------
    np.ndarray
        envelope
    """
    # 1. Narrowband bandpass
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    narrow = filtfilt(b, a, ecog_signal, axis=0)
    # 2. Hilbert transform to get analytic signal
    analytic = hilbert(narrow, axis=0)
    # 3. Envelope = absolute value
    envelope = np.abs(analytic)
    return envelope

def multiband_features(ecog_raw: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    """Builds concatenated band-power features for μ, β, and high-gamma.
    Parameters
    ----------
    ecog_raw : np.ndarray
        (T, 64)
    fs : float, optional
        Frequency of the sample, by default 1000.0
    Returns
    -------
    np.ndarray
        features: (T, 64, 3) (μ, β, high-gamma per electrode)
    """
    mu_env = band_power_envelope(ecog_raw, lowcut=8.0, highcut=13.0, fs=fs)
    beta_env = band_power_envelope(ecog_raw, lowcut=13.0, highcut=30.0, fs=fs)
    hg_env = band_power_envelope(ecog_raw, lowcut=70.0, highcut=200.0, fs=fs)
    # Concatenate along channel dimension
    return np.concatenate([mu_env, beta_env, hg_env], axis=1)


def create_overlapping_windows(ecog_values: np.ndarray, motion_values: np.ndarray, window_size: int = 20, hop_size: int = 10):
    """Builds overlapping windows to increase sample count and capture smoother transitions.

    Parameters
    ----------
    ecog_values : np.ndarray
        (T, features)
    motion_values : np.ndarray
        (T_motion, 3)_
    window_size : int, optional
        number of timepoints per window, by default 20
    hop_size : int, optional
        step bewteen windows, by default 10
    """
    num_samples, num_features = ecog_values.shape
    max_windows = (num_samples - window_size) // hop_size + 1
    X_list = []
    y_list = []
    for w in range(max_windows):
        start = w * hop_size
        end = start + window_size
        if end > num_samples:
            break
        # Assign label as motion at center of window (or last timepoint)
        X_list.append(ecog_values[start:end, :])
        y_list.append(motion_values[min(end -1, motion_values.shape[0] -1), :])
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y

class PreprocessData:
    def __init__(self, ecog_file_path, motion_file_path):
        self.ecog_file_path = ecog_file_path
        self.motion_file_path = motion_file_path
        self.ecog_data = None
        self.motion_data = None
        self.filtered_ecog = None
        self.scaled_ecog = None
        self.X = None
        self.y = None
        self.scaler = None

    def process(self, eval=False, window_size=20, duration_limit=900):
        self.read_data()
        self.common_average_reference()
        self.filter_signal(eval=eval)
        self.format_data(window_size=window_size, duration_limit=duration_limit)
        return self.X, self.y
    
    def read_data(self):
        self.ecog_data = pd.read_csv(self.ecog_file_path)
        self.motion_data = pd.read_csv(self.motion_file_path)
        return self

    def common_average_reference(self):
        # Subtract the common mean from the signals 
        common_average_reference = np.mean(self.ecog_data.drop(["Time", "Fs"], axis=1).values, axis=1, keepdims=1)
        ecog_data_values = self.ecog_data[self.ecog_data.columns[1:-1]].values
        ecog_data_common_mean_subtracted = ecog_data_values - common_average_reference
        self.ecog_data[self.ecog_data.columns[1:-1]] = ecog_data_common_mean_subtracted
        del ecog_data_values, ecog_data_common_mean_subtracted, common_average_reference
        gc.collect()
        return self

    def filter_signal(self, eval=False):
        ecog_raw = self.ecog_data[self.ecog_data.columns[1:-1]].values

        # Apply filters
        filtered = bandpass_filter(ecog_raw, lowcut=1.0, highcut=200.0, fs=1000.0, order=4)
        denoised = notch_filter(filtered, freq=60, fs=1000.0)

        # Evaluate filters
        if eval:
            kurt_raw = kurtosis(ecog_raw, axis=0, fisher=True)
            kurt_denoised = kurtosis(denoised, axis=0, fisher=True)
            proportion_of_positive_kurtosis_signals(kurt_raw, kurt_denoised)
            compute_rmse(ecog_raw, denoised)

        # Compute Power Envelopes
        features = multiband_features(denoised, fs=1000.0) # shape (T, 192)

        # Identify the principal components of the network
        pca = PCA(n_components = 64, random_state=42)
        reduced = pca.fit_transform(features)

        # Scale
        self.scaler = StandardScaler()
        self.scaled_ecog = self.scaler.fit_transform(reduced)

        # Replace in DataFrame
        self.ecog_data = self.ecog_data.copy()
        self.ecog_data[self.ecog_data.columns[1:-1]] = self.scaled_ecog

        # Clean memory
        del ecog_raw, filtered, denoised
        gc.collect()
        return self

    def format_data(self, window_size=20, duration_limit=900):
        ecog_df = self.ecog_data[self.ecog_data["Time"] <= duration_limit]
        motion_df = self.motion_data[self.motion_data["Motion_time"] <= duration_limit]

        ecog_values = ecog_df.drop(columns=["Fs", "Time"]).values
        motion_values = motion_df.drop(columns=["Fsm", "Motion_time"]).values

        print(f"motion_values.shape: {motion_values.shape}")

        # Smooth the signal
        X, y = create_overlapping_windows(ecog_values, motion_values, window_size=20, hop_size=10)
        print(f"y.shape: {y.shape}")
        self.X, self.y = X, y
        
        print(self.X.shape)
        print(self.y.shape)
        
        # Clean up
        del ecog_values, motion_values
        gc.collect()

    def save(self):
        output_file_path_base = self.ecog_file_path.strip("ecog_data.csv")
        joblib.dump(self.scaler, output_file_path_base + "scaler_ecog.pkl")
        np.save(output_file_path_base + "X.npy", self.X)
        np.save(output_file_path_base + "y.npy", self.y)
