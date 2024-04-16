import os
import mne
import numpy as np
import torch
# first version
# reading all .edf and summary.txt file in the repository.
# setting window duration and overlapping.
# labeling each EEG_data windows(seizure:1/non-seizure:0).
# return value is data_labels_tensor that is list of two tensor(EEG data, label)
def label_seizure_windows(edf_dir, summary_file):
    # .edf file directory
    edf_files = [file for file in os.listdir(edf_dir) if file.endswith('.edf')]
    # reading seizure information from summary.txt and store to seizure_info
    seizure_info = []
    with open(summary_file, 'r') as file:
        filename = None
        for line in file:
            if line.startswith("File Name:"):
                filename = line.split(": ")[1].strip()[:-4]
            elif line.startswith("Number of Seizures in File:"):
                num_seizures = int(line.split(": ")[1].strip())
                if num_seizures > 0:
                    for _ in range(num_seizures):
                        start_time_line = file.readline()
                        end_time_line = file.readline()
                        start_time = int(start_time_line.split(": ")[1].split()[0])
                        end_time = int(end_time_line.split(": ")[1].split()[0])
                        seizure_info.append({'filename': filename, 'start_time': start_time, 'end_time': end_time})
    # to store EEG data and label
    data_labels = []

    # reading .edf files and labeling
    for edf_file in edf_files:
        file_path = os.path.join(edf_dir, edf_file) # .edf file path
        raw = mne.io.read_raw_edf(file_path) # reading .edf file
        # setting windows
        sfreq = raw.info['sfreq'] # sampling freq
        window_duration = 2 # window length(second)
        overlap = 1 # overlapping length(second)
        window_samples = int(window_duration * sfreq) # number of samples in the window
        overlap_samples = int(overlap * sfreq) # number of samples in the overlap

        # store EEG data and label by window
        for i in range(0, len(raw.times) - window_samples, window_samples - overlap_samples):
            window_start = raw.times[i]
            window_end = raw.times[i + window_samples]
            # store the extracted EEG data in window_data
            window_data = raw.copy().crop(tmin=window_start, tmax=window_end)

            # checking seizure or non-seizure and information of seizure
            is_seizure = False
            for seizure in seizure_info:
                if seizure['filename'] == edf_file[:-4] and window_start >= seizure['start_time'] and window_end <= seizure['end_time']:
                    is_seizure = True
                    break
            # add data and label pairs to list
            data_labels.append((window_data.get_data(), 1 if is_seizure else 0))

    # convert list to pytorch tensor (probably need to be corrected!!)
    data_labels_tensor = [(torch.tensor(data), torch.tensor(label)) for data, label in data_labels]
    return data_labels_tensor

edf_directory = 'c:\\grad_proj\\chb-mit-scalp-eeg-database-1.0.0\\chb01'
summary_file_path = 'c:\\grad_proj\\chb-mit-scalp-eeg-database-1.0.0\\chb01\\chb01-summary.txt'
data_labels = label_seizure_windows(edf_directory, summary_file_path)

# for checking results
#for i, (data, label) in enumerate(data_labels):
#    print(f"Window {i+1} EEG data shape: {data.shape}, Label: {label}")
print(data_labels)