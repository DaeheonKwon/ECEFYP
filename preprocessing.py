import os
import mne
import numpy as np
import torch
import logging

def band_pass(raw, bands, window_start, window_end):
    # extract data window form raw
    data = raw[:, window_start:window_end][0]
    all_data = [data]    
    # band pass for all bands
    for low, high in bands:
        copied_data = raw.copy().filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=4)
        window_band_data = copied_data[:, window_start:window_end][0]
        all_data.append(window_band_data)
        all_data_np = np.stack(all_data)

    return all_data_np

# reading seizure information from summary.txt and store to seizure_info
def load_seizure_info(file_path):
    seizure_info = {}
    with open(file_path, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break  # end of file -> break while loop
            if line.startswith("File Name:"):
                filename = line.split(": ")[1].strip()
                seizure_info[filename] = []
                continue
            if line.startswith("Number of Seizures in File:"):
                num_seizures = int(line.split(": ")[1])
                if num_seizures == 0: # non-seizure
                    continue
                for _ in range(num_seizures):
                    start_time_line = file.readline()
                    end_time_line = file.readline()
                    start_time = int(start_time_line.split(": ")[1].split()[0])
                    end_time = int(end_time_line.split(": ")[1].split()[0])
                    seizure_info[filename].append((start_time, end_time))

    print(seizure_info) # for debugging     
    return seizure_info

def process_eeg_data(edf_dir, seizure_info, window_duration=2, overlap_duration=1):
    data_labels_tensors = [] # output tensor
    bands = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28)] # bands for BPF
    #list for controlling channel variation
    channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8']
    
    for edf_file in os.listdir(edf_dir):
        if not edf_file.endswith('.edf'):
            continue
        file_path = os.path.join(edf_dir, edf_file)
        raw = mne.io.read_raw_edf(file_path, preload=True)
        raw.resample(64, npad='auto') # down sampling to 64Hz
        # select channels in .edf file
        channels_in_edf_file = []
        for channel in channels:
            if channel in raw.ch_names:
                channels_in_edf_file.append(channel)
        raw.pick(channels_in_edf_file)
        # for T8-P8 duplication case
        if 'T8-P8-0' in raw.ch_names:
            raw.rename_channels({'T8-P8-0': 'T8-P8'})
        #print(raw.ch_names) # for checking channel list in edf file
        print(f"{len(raw.ch_names)} channels in {edf_file}")
        
        sfreq = raw.info['sfreq'] # sampling rate
        window_samples = int(window_duration * sfreq) # sample

        i = 0
        while i <= raw.n_times - window_samples:
            window_start = i
            window_end = i + window_samples

            #print("Channels before extraction:", raw.ch_names)
            data, _ = raw[:, window_start:window_end]
            #print("Shape of extracted data:", data.shape)

            # labeling seizure information
            is_seizure = False
            if seizure_info.get(edf_file):  # if seizures are in .edf file
                for (start_time, end_time) in seizure_info[edf_file]:
                    #print(start_time, end_time)
                    if (start_time < window_end / sfreq <= end_time) or (start_time <= window_start / sfreq < end_time):
                        is_seizure = True
                        break
            # adjust the overlap for the next window
            if is_seizure:
                overlap_samples = int(window_samples * 0.9)  # 90% overlap for seizures
            else:
                overlap_samples = int(window_samples * 0.5)  # 50% overlap for non-seizures
            i += window_samples - overlap_samples  # Move the index forward
            
            # cancel terminal message
            #logging.getLogger('mne').setLevel(logging.WARNING)

            # Processing the EEG data for bandpass filtering
            all_data = [data]  # Include the raw data
            # BPF, bands = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28)]
            #for low, high in bands:
            #    band_data = raw_copy.copy().filter(l_freq=low, h_freq=high, fir_design='firwin')
            #    window_band_data = band_data[:, window_start:window_end][0]
            #    all_data.append(window_band_data)
            
            # numpy
            all_data_np = np.stack(all_data)

            #processed_data = band_pass(raw, bands, window_start, window_end)
            
            data_tensor = torch.tensor(all_data_np, dtype=torch.float32)
            data_tensor = data_tensor.permute(1, 2, 0) # demension control (channel, sample, filtered)
            label_tensor = torch.tensor(int(is_seizure), dtype=torch.int64)
            data_labels_tensors.append((data_tensor, label_tensor))
            #print(data_tensor.shape)
    return data_labels_tensors

# file directory
edf_directory = 'c:\\grad_proj\\chb-mit-scalp-eeg-database-1.0.0\\chb01'
summary_file_path = 'c:\\grad_proj\\chb-mit-scalp-eeg-database-1.0.0\\chb01\\chb01-summary.txt'

seizure_info = load_seizure_info(summary_file_path)
data_labels = process_eeg_data(edf_directory, seizure_info)
print(f"Total windows processed: {len(data_labels)}")