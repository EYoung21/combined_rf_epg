import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import pickle
import optuna
import warnings
import tqdm
from imblearn.over_sampling import BorderlineSMOTE

warnings.simplefilter(action='ignore', category=FutureWarning)

class Model():
    def __init__(self, save_path = None, trial = None):
        # Optimal hyperparameters from Optuna Trial 61 with SMOTE (F1=0.4906)
        self.chunk_seconds = 1
        self.window_size = 20   # of seconds around each chunk

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


    def transform_data(self, probes, training = True):
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
                window_fft = np.abs(fft(signal))[1:n//2 ]
                window_freqs = fftfreq(n, 1 / self.sample_rate)[1:n//2]

                num_largest = self.num_freqs

                # gets top-k indices without fully sorting ~O(n)
                indices = (-window_fft).argpartition(num_largest, axis=None)[:num_largest]
                # sort on that so O(k log k)
                indices = sorted(indices, key=lambda x: window_fft[x], reverse=True)

                peak_freqs = window_freqs[indices]


                # chunk_fft = np.abs(fft(chunk[self.waveform_type].values))[1:self.chunk_size//2]
                # chunk_freqs = fftfreq(self.chunk_size, 1 / self.sample_rate)[1:self.chunk_size//2]
                
                # num_largest = self.num_freqs
                # indices = (-chunk_fft).argpartition(num_largest, axis=None)[:num_largest]
                # indices = sorted(indices, key=lambda x: chunk_fft[x], reverse=True)

                # peak_freqs = chunk_freqs[indices]

                for i in range(num_largest):
                    columns[f"F{i}"].append(peak_freqs[i])
                
                chunk_signal = chunk[self.waveform_type].values
                chunk_time = np.arange(len(chunk_signal)).reshape(-1, 1) 
                lr = LinearRegression()
                lr.fit(chunk_time, chunk_signal)
                slope = lr.coef_[0]  
                
                columns["mean"].append(np.mean(chunk[self.waveform_type]))
                columns["std"].append(np.std(chunk[self.waveform_type]))
                columns["slope"].append(slope)  # Add slope feature
                columns["resistance"].append(chunk["resistance"].values[0])
                columns["volts"].append(chunk["voltage"].values[0])
                columns["current"].append(0 if chunk["current"].values[0] == "AC" else 1)
                if training: # In reality, we won't know what the labels are
                    labels, label_counts = np.unique(chunk["labels"], return_counts=True)
                    label = labels[np.argmax(label_counts)]
                    columns["label"].append(label)

            probe_out = pd.DataFrame(columns)
            transformed_probes.append(probe_out)
        return transformed_probes

    def train(self, probes, test_data, fold):
        transformed_probes = self.transform_data(probes)
        train = pd.concat(transformed_probes)
        X_train = train.drop(["label"], axis=1)
        Y_train = train["label"]

        # initialize BorderlineSMOTE and perform resample
        smote = BorderlineSMOTE(random_state=self.random_state)
        X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

        rf = RandomForestClassifier(self.num_estimators, class_weight="balanced", max_depth = self.max_depth)
        self.model = rf.fit(X_train_resampled, Y_train_resampled)
    
    def predict(self, probes):
        transformed_probes = self.transform_data(probes, training = False)
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

    def load(self, path = None):
        with open(path, 'rb') as model_save:
            self.model = pickle.load(model_save)
