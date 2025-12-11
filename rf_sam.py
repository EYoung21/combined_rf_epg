import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import pickle
import warnings
from scipy.signal import hilbert

warnings.simplefilter(action='ignore', category=FutureWarning)

class Model:
    def __init__(self, save_path=None, trial=None):
        # Hyperparameters
        self.chunk_seconds = 2
        self.num_estimators = 64
        self.max_depth = 8
        self.max_features = 0.7495496906146624  
        self.num_freqs = 9
        self.sample_rate = 100
        self.chunk_size = self.chunk_seconds * self.sample_rate
        self.waveform_type = "post_rect"
        self.random_state = 42
        self.save_path = save_path
        self.model_path = "../ML/rf_pickle"
        self.overlap = 0.4888198026220527
        self.max_lag = 5

        # Optuna tuning
        if trial:
            self.chunk_seconds = trial.suggest_int('chunk_seconds', 1, 4)
            self.num_freqs = trial.suggest_int('num_freqs', 7, 15)
            self.num_estimators = trial.suggest_categorical('num_estimators', [16, 32, 64, 128])
            self.max_depth = trial.suggest_categorical('max_depth', [8, 16, 32, 64, 128])
            self.max_features = trial.suggest_float('max_features', 0.5,1.0)
            self.chunk_size = self.chunk_seconds * self.sample_rate
            self.overlap = trial.suggest_float('overlap', 0.3, 0.8)
            self.max_lag = trial.suggest_int('max_lag', 3,10)

        self.model = None

        # calculate window for weighted voting where emphasis on center
        sigma = self.chunk_size / 6
        center = (self.chunk_size - 1) / 2
        self.window_weight = np.exp(-0.5 * ((np.arange(self.chunk_size) - center) / sigma) ** 2)
        self.window_weight /= np.sum(self.window_weight)

    def transform_data(self, probes, training=True):
        transformed_probes = []

        for probe in probes:
            data_array = probe[self.waveform_type].values
            labels_array = probe["labels"].values if training else None
            n_samples = len(data_array)
            increment = max(1, int(self.chunk_size * (1 - self.overlap)))

            start_indices = np.arange(0, max(n_samples - self.chunk_size + 1, 1), increment)
            n_windows = len(start_indices)

            freqs = np.fft.fftfreq(self.chunk_size, 1 / self.sample_rate)[1:self.chunk_size // 2]
            columns = defaultdict(list)

            # Loop over windows 
            for start in start_indices:
                end = start + self.chunk_size
                if end <= n_samples:
                    window = data_array[start:end]
                    label_window = labels_array[start:end] if training else None
                else:  # pad last window with last val
                    pad_size = end - n_samples
                    window = np.concatenate([data_array[start:], np.full(pad_size, data_array[-1])])
                    if training:
                        label_window = np.concatenate([labels_array[start:], np.full(pad_size, labels_array[-1])])

                # FFT
                fft_vals = np.abs(np.fft.fft(window))[1:self.chunk_size // 2]
                fft_norm = np.linalg.norm(fft_vals) + 1e-12
                fft_vals /= fft_norm
                
                dominant_idx = np.argmax(fft_vals)
                dominant_freq = freqs[dominant_idx]
                columns["dom_freq"].append(dominant_freq)

                # Peak frequencies
                peak_count = min(self.num_freqs, len(fft_vals))
                peak_indices = np.argsort(-fft_vals)[:peak_count]
                peak_freqs = freqs[peak_indices]
                if peak_count < self.num_freqs:
                    peak_freqs = np.pad(peak_freqs, (0, self.num_freqs - peak_count))
                for j in range(self.num_freqs):
                    columns[f"F{j}"].append(peak_freqs[j])

                # Log FFT and delta FFT
                log_fft = np.log1p(fft_vals)
                log_fft = np.pad(log_fft, (0, max(0, self.num_freqs - len(log_fft))))
                for j in range(self.num_freqs):
                    columns[f"logF{j}"].append(log_fft[j])

                # Spectral features
                spectral_power = fft_vals / (np.sum(fft_vals) + 1e-6)
                columns["spec_centroid"].append(np.sum(spectral_power * freqs[:len(fft_vals)]))
                columns["spec_bandwidth"].append(np.sqrt(np.sum(((freqs[:len(fft_vals)] - columns["spec_centroid"][-1]) ** 2) * spectral_power)))
                columns["spec_rolloff"].append(freqs[np.argmax(np.cumsum(spectral_power) >= 0.85)])
                power_norm = spectral_power / (np.sum(spectral_power) + 1e-12)
                columns["spec_entropy"].append(-np.sum(power_norm * np.log2(power_norm + 1e-12)))

                # Envelope features
                envelope = np.abs(hilbert(window))
                columns["env_mean"].append(np.mean(envelope))
                columns["env_std"].append(np.std(envelope))
                columns["env_max"].append(np.max(envelope))

                # Time-domain features
                columns["mean"].append(np.mean(window))
                columns["std"].append(np.std(window))
                columns["range"].append(np.ptp(window))
                columns["rms"].append(np.sqrt(np.mean(window**2)))
                columns["iqr"].append(np.percentile(window, 75) - np.percentile(window, 25))
                columns["zero_crossing"].append(np.sum((window[:-1] * window[1:]) < 0))

                # Autocorrelation
                for lag in range(1, self.max_lag + 1):
                    x = window[:-lag]
                    y = window[lag:]
                    cov = np.mean((x - np.mean(x)) * (y - np.mean(y)))
                    std_x = np.std(x)
                    std_y = np.std(y)
                    columns[f"autocorr_lag{lag}"].append(cov / (std_x * std_y + 1e-12))

                # Label
                if training:
                    vals, counts = np.unique(label_window, return_counts=True)
                    columns["label"].append(vals[np.argmax(counts)])

            # Constant probe info
            columns["resistance"] = [probe["resistance"].values[0]] * n_windows
            columns["volts"] = [probe["voltage"].values[0]] * n_windows
            columns["current"] = [0 if probe["current"].values[0] == "AC" else 1] * n_windows
            columns["position"] = np.arange(n_windows) / max(1, n_windows)

            transformed_probes.append(pd.DataFrame(columns).fillna(0))

        return transformed_probes

    def train(self, probes, test_data=None, fold=None):
        transformed_probes = self.transform_data(probes)
        train = pd.concat(transformed_probes)
        X_train = train.drop(["label"], axis=1)
        Y_train = train["label"]
        self.model = RandomForestClassifier(
            n_estimators=self.num_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            class_weight="balanced",
            random_state=self.random_state
        ).fit(X_train, Y_train)

    def predict(self, probes):
        transformed_probes = self.transform_data(probes, training=False)
        predictions = []

        for df, raw_probe in zip(transformed_probes, probes):
            proba = self.model.predict_proba(df)
            classes = self.model.classes_
            n_samples = len(raw_probe)
            increment = max(1, int(self.chunk_size * (1 - self.overlap)))

            votes = np.zeros((n_samples, len(classes)))
            counts = np.zeros(n_samples)

            # all window start indices
            starts = np.arange(0, n_samples - self.chunk_size + 1, increment)
            for i, start in enumerate(starts):
                end = start + self.chunk_size
                sl = slice(start, end)
                votes[sl] += proba[i] * self.window_weight[:end-start, None]
                counts[sl] += self.window_weight[:end-start]

            # last window
            last_sl = slice(n_samples - self.chunk_size, n_samples)
            votes[last_sl] += proba[-1] * self.window_weight[:self.chunk_size, None]
            counts[last_sl] += self.window_weight[:self.chunk_size]

            # normalize and pick class
            avg_votes = votes / counts[:, None]
            predictions.append(classes[np.argmax(avg_votes, axis=1)])

        return predictions

    def save(self):
        with open(self.model_path, 'ab') as f:
            pickle.dump(self.model, f)

    def load(self, path=None):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
