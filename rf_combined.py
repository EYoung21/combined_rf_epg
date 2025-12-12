"""
Combined Random Forest Model for EPG Signal Classification

This unified model combines the best features from all three RF implementations:
- BorderlineSMOTE for class balancing (from Carson)
- Overlapping windows with weighted voting (from Sam)
- Spectral features: centroid, bandwidth, rolloff, entropy (from Sam)
- Envelope features via Hilbert transform (from Sam)
- Autocorrelation features (from Sam)
- Subwindow analysis (from Subwindow)
- Slope/trend features (from Carson/Subwindow)
"""

from typing import Any
import os
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import BorderlineSMOTE
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class Model:
    """
    Unified Random Forest model combining features from all three implementations.
    """
    
    def __init__(self, save_path=None, trial=None):
        # Core hyperparameters
        self.chunk_seconds = 1
        self.sample_rate = 100
        self.chunk_size = self.chunk_seconds * self.sample_rate
        self.window_seconds = 8
        self.window_size = self.window_seconds * self.sample_rate
        
        # RF hyperparameters
        self.num_estimators = 128
        self.max_depth = 32
        self.max_features = 0.75
        self.random_state = 42
        
        # Feature extraction parameters
        self.num_freqs = 9
        self.waveform_type = "post_rect"
        self.overlap = 0.5  # Overlapping windows for prediction
        self.max_lag = 5  # Autocorrelation lags
        
        # Subwindow parameters
        self.num_subwindows = 4
        self.subwindow_freq = 8
        self.subwindow_size = int(self.window_size / self.num_subwindows)
        
        # Feature toggles
        self.use_slope = True
        self.use_spectral = True
        self.use_envelope = True
        self.use_autocorr = True
        self.use_subwindows = True
        
        self.model = None
        self.save_path = save_path
        self.model_path = "../ML/rf_pickle"
        
        # Optuna tuning
        if trial:
            self.chunk_seconds = trial.suggest_int('chunk_seconds', 1, 4)
            self.window_seconds = trial.suggest_int('window_seconds', 4, 12)
            self.num_freqs = trial.suggest_int('num_freqs', 5, 15)
            self.num_estimators = trial.suggest_categorical('num_estimators', [64, 128, 256])
            self.max_depth = trial.suggest_categorical('max_depth', [16, 32, 64])
            self.max_features = trial.suggest_float('max_features', 0.5, 1.0)
            self.overlap = trial.suggest_float('overlap', 0.3, 0.7)
            self.max_lag = trial.suggest_int('max_lag', 3, 8)
            self.num_subwindows = trial.suggest_int('num_subwindows', 2, 6)
            self.subwindow_freq = trial.suggest_int('subwindow_freq', 4, 12)
            
            # Recalculate derived values
            self.chunk_size = self.chunk_seconds * self.sample_rate
            self.window_size = self.window_seconds * self.sample_rate
            self.subwindow_size = int(self.window_size / self.num_subwindows)
        
        # Calculate window weights for weighted voting (Gaussian centered)
        sigma = self.chunk_size / 6
        center = (self.chunk_size - 1) / 2
        self.window_weight = np.exp(-0.5 * ((np.arange(self.chunk_size) - center) / sigma) ** 2)
        self.window_weight /= np.sum(self.window_weight)

    def _extract_fft_features(self, signal, num_freqs, prefix=""):
        """Extract FFT-based features from a signal."""
        n = len(signal)
        if n < 4:
            # Return zeros if signal too short
            features = {}
            for i in range(num_freqs):
                features[f"{prefix}F{i}"] = 0.0
                features[f"{prefix}logF{i}"] = 0.0
            features[f"{prefix}dom_freq"] = 0.0
            if self.use_spectral:
                features[f"{prefix}spec_centroid"] = 0.0
                features[f"{prefix}spec_bandwidth"] = 0.0
                features[f"{prefix}spec_rolloff"] = 0.0
                features[f"{prefix}spec_entropy"] = 0.0
            return features
        
        fft_vals = np.abs(fft(signal))[1:n//2]
        freqs = fftfreq(n, 1 / self.sample_rate)[1:n//2]
        
        if len(fft_vals) == 0:
            features = {}
            for i in range(num_freqs):
                features[f"{prefix}F{i}"] = 0.0
                features[f"{prefix}logF{i}"] = 0.0
            features[f"{prefix}dom_freq"] = 0.0
            if self.use_spectral:
                features[f"{prefix}spec_centroid"] = 0.0
                features[f"{prefix}spec_bandwidth"] = 0.0
                features[f"{prefix}spec_rolloff"] = 0.0
                features[f"{prefix}spec_entropy"] = 0.0
            return features
        
        # Normalize FFT
        fft_norm = np.linalg.norm(fft_vals) + 1e-12
        fft_normalized = fft_vals / fft_norm
        
        features = {}
        
        # Dominant frequency
        dominant_idx = np.argmax(fft_normalized)
        features[f"{prefix}dom_freq"] = freqs[dominant_idx]
        
        # Peak frequencies
        peak_count = min(num_freqs, len(fft_normalized))
        peak_indices = np.argsort(-fft_normalized)[:peak_count]
        peak_freqs = freqs[peak_indices]
        
        for i in range(num_freqs):
            if i < len(peak_freqs):
                features[f"{prefix}F{i}"] = peak_freqs[i]
            else:
                features[f"{prefix}F{i}"] = 0.0
        
        # Log FFT features
        log_fft = np.log1p(fft_normalized)
        for i in range(num_freqs):
            if i < len(log_fft):
                features[f"{prefix}logF{i}"] = log_fft[peak_indices[i]] if i < len(peak_indices) else 0.0
            else:
                features[f"{prefix}logF{i}"] = 0.0
        
        # Spectral features
        if self.use_spectral:
            spectral_power = fft_normalized / (np.sum(fft_normalized) + 1e-12)
            freq_subset = freqs[:len(fft_normalized)]
            
            # Spectral centroid
            centroid = np.sum(spectral_power * freq_subset)
            features[f"{prefix}spec_centroid"] = centroid
            
            # Spectral bandwidth
            features[f"{prefix}spec_bandwidth"] = np.sqrt(np.sum(((freq_subset - centroid) ** 2) * spectral_power))
            
            # Spectral rolloff (85%)
            cumsum = np.cumsum(spectral_power)
            rolloff_idx = np.argmax(cumsum >= 0.85)
            features[f"{prefix}spec_rolloff"] = freq_subset[rolloff_idx] if rolloff_idx < len(freq_subset) else 0.0
            
            # Spectral entropy
            power_norm = spectral_power / (np.sum(spectral_power) + 1e-12)
            features[f"{prefix}spec_entropy"] = -np.sum(power_norm * np.log2(power_norm + 1e-12))
        
        return features

    def _extract_time_features(self, signal, prefix=""):
        """Extract time-domain features from a signal."""
        features = {}
        
        features[f"{prefix}mean"] = np.mean(signal)
        features[f"{prefix}std"] = np.std(signal)
        features[f"{prefix}range"] = np.ptp(signal)
        features[f"{prefix}rms"] = np.sqrt(np.mean(signal**2))
        features[f"{prefix}iqr"] = np.percentile(signal, 75) - np.percentile(signal, 25)
        features[f"{prefix}zero_crossing"] = np.sum((signal[:-1] * signal[1:]) < 0) if len(signal) > 1 else 0
        
        # Slope via linear regression
        if self.use_slope and len(signal) > 1:
            time_steps = np.arange(len(signal)).reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(time_steps, signal)
            features[f"{prefix}slope"] = lr.coef_[0]
        
        # Envelope features via Hilbert transform
        if self.use_envelope and len(signal) > 1:
            envelope = np.abs(hilbert(signal))
            features[f"{prefix}env_mean"] = np.mean(envelope)
            features[f"{prefix}env_std"] = np.std(envelope)
            features[f"{prefix}env_max"] = np.max(envelope)
        
        # Autocorrelation features
        if self.use_autocorr:
            for lag in range(1, self.max_lag + 1):
                if len(signal) > lag:
                    x = signal[:-lag]
                    y = signal[lag:]
                    cov = np.mean((x - np.mean(x)) * (y - np.mean(y)))
                    std_x = np.std(x)
                    std_y = np.std(y)
                    features[f"{prefix}autocorr_lag{lag}"] = cov / (std_x * std_y + 1e-12)
                else:
                    features[f"{prefix}autocorr_lag{lag}"] = 0.0
        
        return features

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
                # Create window around chunk
                extra_context = (self.window_size - self.chunk_size) // 2
                chunk_start = i * self.chunk_size
                window_start = max(0, chunk_start - extra_context)
                window_end = min(len(probe), chunk_start + self.chunk_size + extra_context)
                window = probe.iloc[window_start:window_end]
                
                window_signal = window[self.waveform_type].values
                chunk_signal = chunk[self.waveform_type].values
                
                # Extract FFT features from window
                fft_features = self._extract_fft_features(window_signal, self.num_freqs)
                for key, val in fft_features.items():
                    columns[key].append(val)
                
                # Extract time-domain features from chunk
                time_features = self._extract_time_features(chunk_signal)
                for key, val in time_features.items():
                    columns[key].append(val)
                
                # Subwindow features
                if self.use_subwindows:
                    subwindows = np.array_split(window, self.num_subwindows)
                    for idx, subwindow in enumerate(subwindows):
                        sub_signal = subwindow[self.waveform_type].values
                        
                        # Subwindow FFT features
                        sub_fft = self._extract_fft_features(sub_signal, self.subwindow_freq, prefix=f"sub{idx}_")
                        for key, val in sub_fft.items():
                            columns[key].append(val)
                        
                        # Subwindow time features (simplified)
                        columns[f"sub{idx}_mean"].append(np.mean(sub_signal))
                        columns[f"sub{idx}_std"].append(np.std(sub_signal))
                
                # Probe metadata
                columns["resistance"].append(chunk["resistance"].values[0])
                columns["volts"].append(chunk["voltage"].values[0])
                columns["current"].append(0 if chunk["current"].values[0] == "AC" else 1)
                
                # Labels for training
                if training:
                    labels, label_counts = np.unique(chunk["labels"], return_counts=True)
                    columns["label"].append(labels[np.argmax(label_counts)])
            
            probe_out = pd.DataFrame(columns)
            transformed_probes.append(probe_out)
        
        return transformed_probes

    def train(self, probes, test_data=None, fold=None):
        transformed_probes = self.transform_data(probes)
        train = pd.concat(transformed_probes)
        X_train = train.drop(["label"], axis=1)
        Y_train = train["label"]
        
        # Apply BorderlineSMOTE for class balancing
        smote = BorderlineSMOTE(random_state=self.random_state)
        X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=self.num_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            class_weight="balanced",
            random_state=self.random_state
        ).fit(X_train_resampled, Y_train_resampled)
    
    def predict(self, probes):
        transformed_probes = self.transform_data(probes, training=False)
        predictions = []
        
        for df, raw_probe in zip(transformed_probes, probes):
            if len(df) == 0:
                predictions.append(np.zeros(len(raw_probe), dtype=int))
                continue
            
            # Get class probabilities
            proba = self.model.predict_proba(df)
            classes = self.model.classes_
            n_samples = len(raw_probe)
            increment = max(1, int(self.chunk_size * (1 - self.overlap)))
            
            # Weighted voting across overlapping windows
            votes = np.zeros((n_samples, len(classes)))
            counts = np.zeros(n_samples)
            
            for i in range(len(df)):
                start = i * self.chunk_size
                end = min(start + self.chunk_size, n_samples)
                window_len = end - start
                
                if window_len > 0:
                    weight = self.window_weight[:window_len]
                    votes[start:end] += proba[i] * weight[:, None]
                    counts[start:end] += weight
            
            # Handle any zero counts
            counts = np.maximum(counts, 1e-12)
            
            # Normalize and pick class
            avg_votes = votes / counts[:, None]
            pred = classes[np.argmax(avg_votes, axis=1)]
            predictions.append(pred)
        
        return predictions

    def save(self):
        with open(self.model_path, 'ab') as f:
            pickle.dump(self.model, f)

    def load(self, path=None):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
