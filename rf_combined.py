"""
Combined Random Forest Models for EPG Signal Classification

This file contains three different RF model implementations:
- ModelCarson: Uses BorderlineSMOTE with window-based FFT and slope features
- ModelSam: Uses overlapping windows with spectral/envelope features and weighted voting
- ModelSubwindow: Uses subwindows within windows with optional SMOTE

Each model follows the same interface:
- __init__(save_path=None, trial=None)
- transform_data(probes, training=True)
- train(probes, test_data, fold)
- predict(probes)
- save()
- load(path)
"""

from typing import Any
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
import pickle
import optuna
import warnings
import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)


# =============================================================================
# ModelCarson: BorderlineSMOTE with Window-based FFT
# =============================================================================

class ModelCarson:
    """
    Random Forest model using BorderlineSMOTE for class balancing.
    Uses a sliding window around each chunk for FFT analysis.
    Optimal hyperparameters from Optuna Trial 61 with SMOTE (F1=0.4906)
    """
    
    def __init__(self, save_path=None, trial=None):
        self.chunk_seconds = 1
        self.window_size = 20  # of seconds around each chunk

        self.num_estimators = 128
        self.num_freqs = 9
        self.max_depth = 32
        self.sample_rate = 100
        self.waveform_type = "post_rect"
        self.random_state = 42
        dirname = os.path.dirname(__file__)
        self.model = None
        self.save_path = save_path
        self.model_path = "../ML/rf_pickle"
        
        if trial:
            # Based on Optuna results: chunk_seconds=1 performs best, so focus search there
            self.chunk_seconds = trial.suggest_int('chunk_seconds', 1, 2)
            # Window size: results show preference for larger windows (10+)
            self.window_size = trial.suggest_int("window_size", 8, 30)
            self.num_freqs = trial.suggest_int('num_freqs', 8, 10)
            self.num_estimators = trial.suggest_categorical('num_estimators', [128, 256])
            self.max_depth = trial.suggest_categorical('max_depth', [16, 32])

        self.chunk_size = self.chunk_seconds * self.sample_rate

    def transform_data(self, probes, training=True):
        transformed_probes = []
        for probe in probes:
            num_chunks = len(probe) // self.chunk_size
            if num_chunks == 0:
                print(len(probe))
                print(self.chunk_size)
            chunks = np.array_split(probe[:num_chunks * self.chunk_size], num_chunks)
            columns = defaultdict(list)

            window_samples = int(max(self.window_size * self.sample_rate, self.chunk_size))
            half_extra = (window_samples - self.chunk_size) // 2
            N = len(probe)

            for i, chunk in enumerate(chunks):
                start_chunk = i * self.chunk_size
                end_chunk = start_chunk + self.chunk_size

                # create window
                start_window = max(0, start_chunk - half_extra)
                end_window = min(N, end_chunk + half_extra)
                window = probe.iloc[start_window:end_window]

                signal = window[self.waveform_type].values
                n = len(signal)
                window_fft = np.abs(fft(signal))[1:n//2]
                window_freqs = fftfreq(n, 1 / self.sample_rate)[1:n//2]

                num_largest = self.num_freqs

                # gets top-k indices without fully sorting ~O(n)
                indices = (-window_fft).argpartition(num_largest, axis=None)[:num_largest]
                # sort on that so O(k log k)
                indices = sorted(indices, key=lambda x: window_fft[x], reverse=True)

                peak_freqs = window_freqs[indices]

                for j in range(num_largest):
                    columns[f"F{j}"].append(peak_freqs[j])
                
                chunk_signal = chunk[self.waveform_type].values
                chunk_time = np.arange(len(chunk_signal)).reshape(-1, 1)
                lr = LinearRegression()
                lr.fit(chunk_time, chunk_signal)
                slope = lr.coef_[0]
                
                columns["mean"].append(np.mean(chunk[self.waveform_type]))
                columns["std"].append(np.std(chunk[self.waveform_type]))
                columns["slope"].append(slope)
                columns["resistance"].append(chunk["resistance"].values[0])
                columns["volts"].append(chunk["voltage"].values[0])
                columns["current"].append(0 if chunk["current"].values[0] == "AC" else 1)
                if training:
                    labels, label_counts = np.unique(chunk["labels"], return_counts=True)
                    label = labels[np.argmax(label_counts)]
                    columns["label"].append(label)

            probe_out = pd.DataFrame(columns)
            transformed_probes.append(probe_out)
        return transformed_probes

    def train(self, probes, test_data=None, fold=None):
        transformed_probes = self.transform_data(probes)
        train = pd.concat(transformed_probes)
        X_train = train.drop(["label"], axis=1)
        Y_train = train["label"]

        # initialize BorderlineSMOTE and perform resample
        smote = BorderlineSMOTE(random_state=self.random_state)
        X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

        rf = RandomForestClassifier(self.num_estimators, class_weight="balanced", max_depth=self.max_depth)
        self.model = rf.fit(X_train_resampled, Y_train_resampled)
    
    def predict(self, probes):
        transformed_probes = self.transform_data(probes, training=False)
        predictions = []
        for transformed_probe, raw_probe in zip(transformed_probes, probes):
            test_probe = transformed_probe
            pred = self.model.predict(test_probe)

            # We need to expand the prediction based on the sample rate
            pred = np.repeat(pred, self.chunk_seconds * self.sample_rate)
            # Expand until the end since probe is never exactly divisible by window size
            pred = np.pad(pred, (0, len(raw_probe) - len(pred)), 'edge')
            predictions.append(pred)
        return predictions

    def save(self):
        with open(self.model_path, 'ab') as model_save:
            pickle.dump(self.model, model_save)

    def load(self, path=None):
        with open(path, 'rb') as model_save:
            self.model = pickle.load(model_save)


# =============================================================================
# ModelSam: Overlapping Windows with Spectral/Envelope Features
# =============================================================================

class ModelSam:
    """
    Random Forest model using overlapping windows with weighted voting.
    Features include spectral centroid, bandwidth, rolloff, entropy,
    envelope features via Hilbert transform, and autocorrelation.
    """
    
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
            self.max_features = trial.suggest_float('max_features', 0.5, 1.0)
            self.chunk_size = self.chunk_seconds * self.sample_rate
            self.overlap = trial.suggest_float('overlap', 0.3, 0.8)
            self.max_lag = trial.suggest_int('max_lag', 3, 10)

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


# =============================================================================
# ModelSubwindow: Subwindows with Optional SMOTE
# =============================================================================

class ModelSubwindow:
    """
    Random Forest model using subwindows within larger windows.
    Extracts features from both the main window and multiple subwindows.
    Supports optional SMOTE for class balancing.
    Optimized from Trial 46: F1=0.4904
    """
    
    def __init__(self, save_path=None, trial=None):
        # chunk hyperparameters - OPTIMIZED FROM BEST TRIAL (Trial 46: F1=0.4904)
        self.chunk_seconds = 1
        self.num_estimators = 128
        self.num_freqs = 7
        self.sample_rate = 100
        self.chunk_size = self.chunk_seconds * self.sample_rate
        self.window_seconds = 8
        self.window_size = self.window_seconds * self.sample_rate

        self.max_depth = 16
        self.waveform_type = "post_rect"
        self.random_state = 42
        dirname = os.path.dirname(__file__)
        self.model = None
        self.save_path = save_path
        self.model_path = "../ML/rf_pickle"

        self.num_subwindows = 4
        self.subwindow_freq = 8
        self.subwindow_size = int(self.window_size / self.num_subwindows)
        
        # Boolean flags for optional slope features
        self.use_window_slope = False
        self.use_subwindow_slope = True
        
        # SMOTE hyperparameters
        self.use_smote = True
        self.smote_k_neighbors = 5
        self.smote_sampling_strategy = 'auto'
        
        if trial:
            self.chunk_seconds = trial.suggest_int('chunk_seconds', 1, 10)
            self.num_freqs = trial.suggest_int('num_freqs', 1, 20)
            self.num_estimators = trial.suggest_categorical('num_estimators', [8, 16, 32, 64, 128, 256])
            self.max_depth = trial.suggest_categorical('max_depth', [4, 8, 16, 32, 64, 128, 256])
            self.window_seconds = trial.suggest_int('window_seconds', 2, 10)
            
            # Recalculate derived values based on trial suggestions
            self.chunk_size = self.chunk_seconds * self.sample_rate
            self.window_size = self.window_seconds * self.sample_rate

            # SUBWINDOW PARAMETER OPTIMIZATION
            self.num_subwindows = trial.suggest_int('num_subwindows', 1, 20)
            self.subwindow_size = int(self.window_size / self.num_subwindows)
            self.subwindow_freq = trial.suggest_int('subwindow_freq', 1, 20)
            
            # Optional slope features
            self.use_window_slope = trial.suggest_categorical('use_window_slope', [True, False])
            self.use_subwindow_slope = trial.suggest_categorical('use_subwindow_slope', [True, False])
            
            # SMOTE hyperparameters
            self.use_smote = trial.suggest_categorical('use_smote', [True, False])
            if self.use_smote:
                self.smote_k_neighbors = trial.suggest_int('smote_k_neighbors', 1, 10)
                self.smote_sampling_strategy = 'auto'

    def transform_data(self, probes, training=True):
        transformed_probes = []
        for probe in probes:
            num_chunks = len(probe) // self.chunk_size
            if num_chunks == 0:
                print(f"Skipping probe: too short (len={len(probe)}, chunk_size={self.chunk_size})")
                continue
            chunks = np.array_split(probe[:num_chunks * self.chunk_size], num_chunks)

            columns = defaultdict(list)
            for i, chunk in enumerate(chunks):
                extra_context_size = (self.window_size - self.chunk_size) // 2

                chunkStartIndex = i * self.chunk_size
                window_start = max(0, chunkStartIndex - extra_context_size)
                window_end = min(len(probe), chunkStartIndex + self.chunk_size + extra_context_size)
                currWindow = probe[window_start:window_end]

                chunk_fft = np.abs(fft(currWindow[self.waveform_type].values))[1:self.window_size//2]
                chunk_freqs = fftfreq(self.window_size, 1 / self.sample_rate)[1:self.window_size//2]

                # Cap num_largest to available frequency bins
                num_largest = min(self.num_freqs, len(chunk_fft) - 1) if len(chunk_fft) > 0 else 0
                if num_largest > 0:
                    indices = (-chunk_fft).argpartition(num_largest, axis=None)[:num_largest]
                else:
                    indices = []
                indices = sorted(indices, key=lambda x: chunk_fft[x], reverse=True)

                peak_freqs = chunk_freqs[indices] if len(indices) > 0 else np.array([])

                # Add frequency features (pad with 0 if fewer frequencies than requested)
                for j in range(num_largest):
                    if j < len(peak_freqs):
                        columns[f"F{j}"].append(peak_freqs[j])
                    else:
                        columns[f"F{j}"].append(0.0)
                        
                columns["mean"].append(np.mean(currWindow[self.waveform_type]))
                columns["std"].append(np.std(currWindow[self.waveform_type]))
                
                # Fit linear regression to capture signal trend (optional)
                if self.use_window_slope:
                    window_signal = currWindow[self.waveform_type].values
                    time_steps = np.arange(len(window_signal)).reshape(-1, 1)
                    lr = LinearRegression()
                    lr.fit(time_steps, window_signal)
                    columns["slope"].append(lr.coef_[0])
                    columns["trend_intercept"].append(lr.intercept_)
                
                columns["resistance"].append(currWindow["resistance"].values[0])
                columns["volts"].append(currWindow["voltage"].values[0])
                columns["current"].append(0 if currWindow["current"].values[0] == "AC" else 1)
                
                subwindows = np.array_split(currWindow, self.num_subwindows)
                for idx, subwindow in enumerate(subwindows):
                    # Calculate all the same features for each subwindow
                    sub_fft = np.abs(fft(subwindow[self.waveform_type].values))[1:self.subwindow_size//2]
                    sub_freqs = fftfreq(self.subwindow_size, 1 / self.sample_rate)[1:self.subwindow_size//2]
                   
                    # Cap num_largest to available frequency bins
                    sub_num_largest = min(self.subwindow_freq, len(sub_fft) - 1) if len(sub_fft) > 0 else 0
                    if sub_num_largest > 0:
                        sub_indices = (-sub_fft).argpartition(sub_num_largest, axis=None)[:sub_num_largest]
                    else:
                        sub_indices = []
                    
                    sub_indices = sorted(sub_indices, key=lambda x: sub_fft[x], reverse=True)
                    sub_peak_freqs = sub_freqs[sub_indices] if len(sub_indices) > 0 else np.array([])

                    # Add frequency features (pad with 0 if fewer frequencies than requested)
                    for j in range(sub_num_largest):
                        if j < len(sub_peak_freqs):
                            columns[f"subwindow_{idx}_F{j}"].append(sub_peak_freqs[j])
                        else:
                            columns[f"subwindow_{idx}_F{j}"].append(0.0)
                            
                    columns[f"subwindow_{idx}_mean"].append(np.mean(subwindow[self.waveform_type]))
                    columns[f"subwindow_{idx}_std"].append(np.std(subwindow[self.waveform_type]))
                    
                    # Fit linear regression to capture signal trend (optional)
                    if self.use_subwindow_slope:
                        sub_signal = subwindow[self.waveform_type].values
                        sub_time_steps = np.arange(len(sub_signal)).reshape(-1, 1)
                        lr = LinearRegression()
                        lr.fit(sub_time_steps, sub_signal)
                        columns[f"subwindow_{idx}_slope"].append(lr.coef_[0])
                        columns[f"subwindow_{idx}_trend_intercept"].append(lr.intercept_)
                    
                    columns[f"subwindow_{idx}_resistance"].append(subwindow["resistance"].values[0])
                    columns[f"subwindow_{idx}_volts"].append(subwindow["voltage"].values[0])
                    columns[f"subwindow_{idx}_current"].append(0 if subwindow["current"].values[0] == "AC" else 1)
                
                if training:
                    labels, label_counts = np.unique(currWindow["labels"], return_counts=True)
                    label = labels[np.argmax(label_counts)]
                    columns["label"].append(label)

            probe_out = pd.DataFrame(columns)
            transformed_probes.append(probe_out)
        return transformed_probes

    def train(self, probes, test_data=None, fold=None):
        transformed_probes = self.transform_data(probes)
        train = pd.concat(transformed_probes)
        X_train = train.drop(["label"], axis=1)
        Y_train = train["label"]
        
        # Apply SMOTE if enabled
        if self.use_smote:
            smote = SMOTE(
                k_neighbors=self.smote_k_neighbors,
                sampling_strategy=self.smote_sampling_strategy,
                random_state=self.random_state
            )
            X_train, Y_train = smote.fit_resample(X_train, Y_train)
        
        rf = RandomForestClassifier(self.num_estimators, class_weight="balanced", max_depth=self.max_depth)
        self.model = rf.fit(X_train, Y_train)
    
    def predict(self, probes):
        transformed_probes = self.transform_data(probes, training=False)
        predictions = []
        for transformed_probe, raw_probe in zip(transformed_probes, probes):
            # Handle probes that were too short and skipped during transform_data
            if len(transformed_probe) == 0:
                pred = np.zeros(len(raw_probe), dtype=int)
                predictions.append(pred)
                continue
            
            test_probe = transformed_probe
            pred = self.model.predict(test_probe)

            # Expand the prediction based on the sample rate
            pred = np.repeat(pred, self.chunk_seconds * self.sample_rate)
            # Expand until the end since probe is never exactly divisible by window size
            pad_length = len(raw_probe) - len(pred)
            if pad_length > 0:
                pred = np.pad(pred, (0, pad_length), 'edge')
            elif pad_length < 0:
                pred = pred[:len(raw_probe)]
            predictions.append(pred)
        return predictions

    def save(self):
        with open(self.model_path, 'ab') as model_save:
            pickle.dump(self.model, model_save)

    def load(self, path=None):
        with open(path, 'rb') as model_save:
            self.model = pickle.load(model_save)


# =============================================================================
# Default Model alias (for backwards compatibility)
# =============================================================================

# Use ModelCarson as the default Model class
Model = ModelCarson

