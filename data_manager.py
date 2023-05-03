import os
import pickle
import copy
import itertools
import random
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
import mne

from patient import Patient
from misc import convert_to_timedelta

SIENA_SCALP_DATASET_PATH = './data/physionet.org/files/siena-scalp-eeg/1.0.0'
CHB_MIT_DATASET_PATH = './data/physionet.org/files/chbmit/1.0.0'

class DataManager:


    def __init__(self):
        self.patients = []
        self.seizures = []


    def load_chb_mit(self):
        ######## LOAD PATIENT DATA ########
        for f in os.listdir(CHB_MIT_DATASET_PATH):
            if f[:3] == 'chb':
                patient = Patient(f, dataset='CHBMIT')

                # Parse thru .edf files first
                for file_name in os.listdir(f'{CHB_MIT_DATASET_PATH}/{f}'):
                    if file_name[-4:] == '.edf':
                        patient.add_edf_file(file_name)
                        # All files are 256Hz for CHB-MIT
                        patient.add_rec_frequency(file_name, 256)

                        patient.edf_files[file_name]['file_path'] = f'{CHB_MIT_DATASET_PATH}/{f}/{file_name}'
                        patient.edf_files[file_name]['patient_name'] = f
                        patient.edf_files[file_name]['seizures'] = []

                for file_name in os.listdir(f'{CHB_MIT_DATASET_PATH}/{f}'):
                    if file_name[-4:] == '.txt': # Parse summary.txt
                        self.parse_chbmit_txt(patient, f'{CHB_MIT_DATASET_PATH}/{f}/{file_name}')
                    else:
                        patient.add_extra_file(file_name)
                self.patients.append(patient)
                

    def parse_chbmit_txt(self, patient, file):
        f = open(file)
        all_text = f.read()
        splitted_text = all_text.split('\n\n')
        splitted_text = [s.strip() for s in splitted_text]
        splitted_text = list(filter(lambda s: s != '', splitted_text))

        current_channel_map = None
        for component in splitted_text:
            print(component)
            print('---')
            # Parse the seizure
            if component.startswith('File'):
                edf_file_name, seizures = self.parse_chbmit_seizure(patient, component, channel_map)
                print(current_channel_map)
                patient.add_channel_map(edf_file_name, current_channel_map)
                for seizure in seizures:
                    patient.add_seizure(edf_file_name, seizure)
                    self.seizures.append(seizure)
                    # patient.rec_times[seizure['file_name']] = seizure['recording_time']
            # Parse the channel map
            elif component.startswith('Channels'):
                channel_map = self.parse_chbmit_channels(component)
                current_channel_map = channel_map


    def parse_chbmit_seizure(self, patient, seizure_txt, channel_map):
        lines = seizure_txt.split('\n')
        sleep = 'sleep' in seizure_txt

        seizures = []

        for line in lines:
            print(line)
            print(';;;;')
            '''
            if line.startswith('File Start Time'):
                # reg_start_time = line.split(':')[-1].strip()
                rec_start_time = '.'.join(line.split(':')[-3:]).strip()
                rec_start_time_delta = convert_to_timedelta(rec_start_time)
            elif line.startswith('File End Time'):
                # reg_end_time = line.split(':')[-1].strip()
                rec_end_time = '.'.join(line.split(':')[-3:]).strip()
                rec_end_time_delta = convert_to_timedelta(rec_end_time)
            '''
            if line.startswith('File Name'):
                file_name = line.split(':')[-1].strip()
            elif line.startswith('Seizure') or line.startswith('Start time'):
                # Parse seizure
                label_str, time_str = line.split(':')

                if len(label_str.split(' ')) == 3:
                    id_str = '1'
                    _, time, _ = label_str.split(' ')
                elif len(label_str.split(' ')) == 4:
                    _, id_str, time, _ = label_str.split(' ')

                seizure_number = int(id_str)
                if time == 'Start':
                    seizure_start_time = int(time_str.strip().split(' ')[0].strip())
                elif time == 'End':
                    seizure_end_time = int(time_str.strip().split(' ')[0].strip())

                    seizure_time = seizure_end_time - seizure_start_time

                    seizure = {
                        'dataset': 'CHBMIT',
                        'number': seizure_number,
                        'file_name': f'{CHB_MIT_DATASET_PATH}/{patient.name}/{file_name}',
                        'start_time': seizure_start_time,
                        'seizure_time': seizure_time,
                        'end_time': seizure_end_time,
                        'sleep': sleep,
                    }

                    seizures.append(seizure)

        # patient.add_rec_time(file_name, rec_start_time_delta, rec_start_time_delta, recording_time)

        return file_name, seizures


    def parse_chbmit_channels(self, channels_txt):
        channel_lines = channels_txt.split('\n')[2:]
        channel_map = dict()
        for line in channel_lines:
            left, right = line.split(':')
            channel_num = int(left.split(' ')[-1])
            channel_string = right.strip()
            channel_map[channel_num] = channel_string
        
        return channel_map
        

    def load_siena_scalp(self):
        ######## LOAD PATIENT DATA ########
        for f in os.listdir(SIENA_SCALP_DATASET_PATH):
            # Patient files
            if f[:2] == 'PN':
                patient = Patient(f, dataset='SIENA')
                for file_name in os.listdir(f'{SIENA_SCALP_DATASET_PATH}/{f}'):
                    # Add edf files to patient
                    if file_name[-4:] == '.edf':
                        patient.add_edf_file(file_name)
                        patient.edf_files[file_name]['file_path'] = f'{SIENA_SCALP_DATASET_PATH}/{f}/{file_name}'
                        patient.edf_files[file_name]['patient_name'] = f
                        patient.edf_files[file_name]['seizures'] = []

                for file_name in os.listdir(f'{SIENA_SCALP_DATASET_PATH}/{f}'):
                    if file_name[-4:] == '.txt':
                        self.parse_siena_txt(patient, f'{SIENA_SCALP_DATASET_PATH}/{f}/{file_name}')
                    else:
                        patient.add_extra_file(file_name)
                self.patients.append(patient)

        ######## LOAD PATIENT CSV ########
        csv_file = open(f'{SIENA_SCALP_DATASET_PATH}/subject_info.csv')
        for line in csv_file.readlines()[1:-1]:
            name, age, gender, seizure, localization, lateralization, num_channels, num_seizures, total_rec_time = line.split(',')
            for patient in self.patients:
                if patient.name == name:
                    patient.set_age(int(age))
                    patient.set_gender(gender)
                    patient.set_seizure(seizure)
                    patient.set_localization(localization)
                    patient.set_lateralization(lateralization)
                    patient.set_num_channels(int(num_channels))
                    patient.set_num_seizures(int(num_seizures))
                    patient.set_total_rec_time(int(total_rec_time))
                

    def plot_seizure_lengths(self):
        seizure_times = []
        for patient in self.patients:
            for seizure in patient.seizures:
                seizure_times.append(seizure['seizure_time'].seconds)
        plt.hist(seizure_times, bins='auto')
        plt.xlabel('Seizure length (seconds)')
        plt.ylabel('Count')
        plt.title('Seizure Lengths')
        plt.savefig('./figs/seizure_length_hist.png')


    def parse_siena_txt(self, patient, txt_file):
        # print(file)
        f = open(txt_file)
        all_text = f.read()
        splitted_text = all_text.split('\n\n')
        splitted_text = [s.strip() for s in splitted_text]
        splitted_text = list(filter(lambda s: s != '', splitted_text))

        print(patient.name)

        current_channel_map = None
        for component in splitted_text:
            if component.startswith('Seizure'):
                edf_file_name, seizure = self.parse_siena_seizure(patient, component)
                # Recording frequency for Siena Scalp is 512Hz for all 
                patient.add_rec_frequency(edf_file_name, 512)
                patient.add_channel_map(edf_file_name, current_channel_map)
                self.seizures.append(seizure)
            elif component.startswith('Channels'):
                channel_map = self.parse_siena_channels(component)
                current_channel_map = channel_map

        # return edf_file_name, seizure


    def parse_siena_channels(self, channels_txt):
        channel_lines = channels_txt.split('\n')[1:]
        channel_map = dict()
        for line in channel_lines:
            left, right = line.split(':')
            channel_num = int(left.split(' ')[-1])
            channel_string = right.strip()
            channel_map[channel_num] = channel_string
        
        return channel_map

    def parse_siena_seizure(self, patient, seizure_txt):
        lines = seizure_txt.split('\n')
        seizure_number = int(lines[0].split(' ')[2])
        sleep = 'sleep' in seizure_txt

        for line in lines:
            if line.startswith('Seizure start time') or line.startswith('Start time'):
                seizure_start_time = line.split(':')[-1].strip()
            elif line.startswith('Seizure end time') or line.startswith('End time'):
                seizure_end_time = line.split(':')[-1].strip()
            elif line.startswith('Registration start time'):
                reg_start_time = line.split(':')[-1].strip()
            elif line.startswith('Registration end time'):
                reg_end_time = line.split(':')[-1].strip()
            elif line.startswith('File name'):
                file_name = line.split(':')[-1].strip()

        seizure_start_time_delta = convert_to_timedelta(seizure_start_time)
        seizure_end_time_delta = convert_to_timedelta(seizure_end_time)
        rec_start_time_delta = convert_to_timedelta(reg_start_time)
        rec_end_time_delta = convert_to_timedelta(reg_end_time)
        recording_time = rec_end_time_delta - rec_start_time_delta
        seizure_time = seizure_end_time_delta - seizure_start_time_delta

        seizure = {
            'number': seizure_number,
            'dataset': 'SIENA',
            'file_name': f'{SIENA_SCALP_DATASET_PATH}/{patient.name}/{file_name}',
            'start_time': seizure_start_time_delta,
            'seizure_time': seizure_time,
            'end_time': seizure_end_time_delta,
            'sleep': sleep,
        }

        patient.add_rec_time(file_name, rec_start_time_delta, rec_start_time_delta, recording_time)
        patient.add_seizure(file_name, seizure)

        return file_name, seizure         
        
    # Epoch length: seconds
    # Preictal length: seconds
    def annotate_chbmit_dataset(self, epoch_length=2, preictal_length=600):
        data = {}
        for patient in self.patients:
            if patient.dataset == 'CHBMIT':
                data[patient.name] = dict()
                for edf_file in patient.edf_files:

                    print(f'=========\n{edf_file}\n========')

                    if 'channel_map' not in patient.edf_files[edf_file]:
                        # patient.edf_files.pop(edf_file)
                        continue

                    # Set up data structure 
                    data[patient.name][edf_file] = dict()

                    # Read raw EDF file
                    edf_file_path = patient.edf_files[edf_file]['file_path']
                    raw = mne.io.read_raw_edf(edf_file_path)

                    # Mark the bad channels
                    bads = []
                    for channel in patient.edf_files[edf_file]['channel_map']:
                        if patient.edf_files[edf_file]['channel_map'][channel] in '-.':
                            bads.append(raw.info.ch_names[channel-1])
                    raw.info['bads'] = bads

                    # Standardize the data
                    eeg_data = raw.get_data()
                    mean = np.mean(eeg_data, axis=1, keepdims=True)
                    std = np.std(eeg_data, axis=1, keepdims=True)
                    raw._data = (eeg_data - mean) / std

                    # Create the epoch events 
                    events = mne.make_fixed_length_events(raw, start=0, stop=None, duration=epoch_length)

                    tmax = epoch_length - (1 / raw.info['sfreq'])

                    # Create the actual epoch object
                    epochs = mne.Epochs(raw, events, tmin=0, tmax=tmax, baseline=None)
                    epochs.drop_bad() # Drops "bad" epochs (mostly beginning and end)

                    # LABELS: 
                    # 0 - interictal
                    # 1 - preictal
                    # 2 - ictal
                    # Create seizure intervals
                    if len(patient.edf_files[edf_file]['seizures']) == 0:
                        # If no seizures, all epochs are interictal
                        labels = [0 for _ in range(len(epochs))]
                        ictal_intervals = []
                        preictal_intervals = []
                    else: 
                        # If seizures, need to check ictal and preictal
                        ictal_intervals = [(seizure['start_time'], seizure['end_time']) for seizure in patient.edf_files[edf_file]['seizures']]

                        preictal_intervals = []
                        for seizure in patient.edf_files[edf_file]['seizures']:
                            if seizure['start_time'] < preictal_length:
                                preictal_intervals.append((0, seizure['start_time'] - 1))
                            else:
                                preictal_intervals.append((seizure['start_time'] - preictal_length, seizure['start_time'] - 1))
                            
                        # Epoch label functions
                        is_ictal = lambda epoch_start: any([epoch_start >= s and epoch_start <= e for s,e in ictal_intervals])
                        is_preictal = lambda epoch_start: any([epoch_start >= s and epoch_start <= e for s,e in preictal_intervals])

                        # Create labels 
                        labels = []
                        for i in range(len(epochs)):
                            epoch_start_sec = epochs.events[i][0] / epochs.info['sfreq']
                            if is_preictal(epoch_start_sec):
                                labels.append(1)
                            elif is_ictal(epoch_start_sec):
                                labels.append(2)
                            else:
                                labels.append(0)

                        '''
                        # Get the data from the raw object
                        channels_to_plot = raw.ch_names[:3]
                        timeseries = raw.get_data(picks=channels_to_plot)

                        # Get the time axis
                        times = np.arange(0, timeseries.shape[1] / raw.info['sfreq'], 1 / raw.info['sfreq'])

                        # Plot the selected channels using Matplotlib
                        fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(15, 6), sharex=True)

                        for idx, (ax, channel, channel_data) in enumerate(zip(axes, channels_to_plot, timeseries)):
                            ax.plot(times, channel_data, label=channel)
                            ax.legend(loc='upper right')
                            ax.set_ylabel('Amplitude (µV)')
                            for s,e in ictal_intervals:
                                ax.axvspan(s, e, color='red', alpha=0.3)
                            for s,e in preictal_intervals:
                                ax.axvspan(s, e, color='yellow', alpha=0.3)

                            if idx == len(channels_to_plot) - 1:
                                ax.set_xlabel('Time (s)')

                        plt.show()
                        print(timeseries.shape)
                        '''

                    # Insert data into data dictionary
                    data[patient.name][edf_file]['labels'] = labels
                    data[patient.name][edf_file]['epochs'] = epochs
                    data[patient.name][edf_file]['ictal_intervals'] = ictal_intervals
                    data[patient.name][edf_file]['preictal_intervals'] = preictal_intervals
        
        self.data = data

        # Write data object to a pickle file
        outfile = f'data_pe{preictal_length}_epoch{epoch_length}.pkl'
        with open(outfile, 'wb') as pf:
            pickle.dump(self.data, pf)

    
    def remove_file_info(self, data):

        num_edf_files = sum([len(data[patient]) for patient in data])
        n_total, n_ictal, n_preictal, n_interictal = self.count_epochs(data)

        for patient in data:
            print(patient)
            all_epochs = [data[patient][file]['epochs'] for file in data[patient]]
            all_epochs = sum([[e for e in epochs] for epochs in all_epochs], [])
            all_labels = [data[patient][file]['labels'] for file in data[patient]]
            all_labels = sum([[l for l in labels] for labels in all_labels], [])
            print(len(all_epochs), len(all_labels))
            data[patient] = {'epochs': all_epochs, 'labels': all_labels}

        return data 
        # print(len(all_epochs), len(all_labels))


    
    def select_channels(self, epochs, channels):
        new_epochs = []
        for e in epochs:
            new_epochs.append(e.get_data(picks=channels)[0])
        return new_epochs
        

    def load_data_from_pkl(self, pkl_file_name):
        with open(pkl_file_name, 'rb') as pf:
            self.data = pickle.load(pf)

    def loocv_splits(self):
        for patient in list(self.data.keys())[:3]:
            print(f'Training on patient: {patient}')
            yield self.get_loocv_split(patient)
    
    def random_5050split(self): 
        if self.data == None:
            raise Error("no data loaded")

        n_test = len(self.data.keys()) // 2
        test_patients = random.sample(list(self.data.keys()), n_test)

        # Testing set
        validation_patients = { patient: self.data[patient] for patient in test_patients }

        # Training set
        training_data = copy.deepcopy(self.data)
        for patient in test_patients: 
            del training_data[patient]

        return validation_patients, training_data



    def train_test_split(self, patients):
        if self.data == None:
            raise Error("no data loaded")

        # Testing set
        validation_patients = { patient: self.data[patient] for patient in patients }

        # Training set
        training_data = copy.deepcopy(self.data)
        for patient in patients: 
            del training_data[patient]

        return validation_patients, training_data
        

    def get_loocv_split(self, patient):
        if self.data == None:
            raise Error("no data loaded")
        
        left_out_patient = { patient: self.data[patient] }
        training_data = copy.deepcopy(self.data)
        del training_data[patient]

        # training_data_balanced = self.get_balanced_sample(training_data)

        # TODO: return formatted data for model training
        return left_out_patient, training_data


    # Assumes "flattened" data -- i.e. no file information
    def get_balanced_obs(self, data, channels, interictal_multiplier=1):
        examples = []
        labels = []
        for patient in data:
            print(patient)
            inter, pre, ictal = self.get_balanced_sample({ patient: data[patient] }, channels, interictal_multiplier=interictal_multiplier)

            examples += inter
            labels += [0 for _ in range(len(inter))]
            examples += pre
            labels += [1 for _ in range(len(pre))]
            examples += ictal
            labels += [2 for _ in range(len(ictal))]

        return examples, labels


    # Assumes "flattened" data -- i.e. no file information
    def get_balanced_sample(self, data, channels, interictal_multiplier=1):
        valid_files = []
        for patient in data:
            for file in data[patient]:
                if all([ch in data[patient][file]['epochs'][0].ch_names for ch in channels]):
                    valid_files.append(file)

        n_total, n_ictal, n_preictal, n_interictal = self.count_epochs(data, valid_files)

        interictal_sample = []
        preictal_sample = []
        ictal_sample = []

        for patient in data:
            all_labels = sum([list(zip(data[patient][file]['labels'], range(len(data[patient][file]['labels'])))) for file in data[patient] if file in valid_files], [])
            all_labels_files = [file for file in data[patient] for _ in range(len(data[patient][file]['labels'])) if file in valid_files]
            zipped = list(zip(all_labels, all_labels_files))

            interictal_indices = random.sample([(og_idx,file) for i,((c,og_idx),file) in enumerate(zipped) if c == 0], n_ictal*interictal_multiplier)
            preictal_indices = random.sample([(og_idx,file) for i,((c,og_idx),file) in enumerate(zipped) if c == 1], n_preictal)
            ictal_indices = random.sample([(og_idx,file) for i,((c,og_idx),file) in enumerate(zipped) if c == 2], n_ictal)

            interictal_sample += [data[patient][file]['epochs'][i] for i,file in interictal_indices]
            preictal_sample += [data[patient][file]['epochs'][i] for i,file in preictal_indices]
            ictal_sample += [data[patient][file]['epochs'][i] for i,file in ictal_indices]

        return interictal_sample, preictal_sample, ictal_sample


    def count_epochs(self, data, valid_files=None):
        total_epochs = 0
        total_ictal = 0
        total_preictal = 0
        total_interictal = 0
        for patient in data:
            for file in data[patient]:
                if valid_files == None or file in valid_files:
                    if 'epochs' in data[patient][file]:
                        num_epochs = len(data[patient][file]['epochs'])
                        num_interictal = data[patient][file]['labels'].count(0)
                        num_preictal = data[patient][file]['labels'].count(1)
                        num_ictal = data[patient][file]['labels'].count(2)

                        total_epochs += num_epochs
                        total_ictal += num_ictal 
                        total_preictal += num_preictal
                        total_interictal += num_interictal

        return total_epochs, total_ictal, total_preictal, total_interictal

    def plot_chb11(self):

        raw_file = CHB_MIT_DATASET_PATH + '/chb11/chb11_99.edf'
        raw = mne.io.read_raw_edf(raw_file)
        timeseries = raw.get_data(picks=['P7-O1', 'P8-O2'])

        ictal_intervals = self.data['chb11']['chb11_99.edf']['ictal_intervals']
        preictal_intervals = self.data['chb11']['chb11_99.edf']['preictal_intervals']

        # Get the time axis
        print(timeseries.shape)
        times = np.arange(0, timeseries.shape[1] / raw.info['sfreq'], 1 / raw.info['sfreq'])

        # Plot the selected channels using Matplotlib
        fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)

        for idx, (ax, channel, channel_data) in enumerate(zip(axes, ['P7-O1', 'P8-O2'], timeseries)):
            ax.plot(times, channel_data, label=channel)
            ax.legend(loc='upper right')
            ax.set_ylabel('Amplitude (µV)')
            for s,e in ictal_intervals:
                ax.axvspan(s, e, color='red', alpha=0.3)
            for s,e in preictal_intervals:
                ax.axvspan(s, e, color='yellow', alpha=0.3)

            if idx == 0:
                ax.set_title('chb11_99.edf Raw EEG Data')
            if idx == 1:
                ax.set_xlabel('Time (s)')

        plt.show()

    def load_edf(self, edf_file, patient):

        raw = mne.io.read_raw_edf(edf_file) 
        print(raw)
        print(raw.info)
        non_eeg_channels = list(filter(lambda name: 'EEG' not in name, raw.info.ch_names))
        raw.info['bads'] = non_eeg_channels
        # print(non_eeg_channels)

        timeseries = raw.get_data(picks=['P7-O1', 'P8-O2'])

        # Get the time axis
        times = np.arange(0, timeseries.shape[1] / raw.info['sfreq'], 1 / raw.info['sfreq'])

        # Plot the selected channels using Matplotlib
        fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)

        for idx, (ax, channel, channel_data) in enumerate(zip(axes, ['P7-O1', ''], timeseries)):
            ax.plot(times, channel_data, label=channel)
            ax.legend(loc='upper right')
            ax.set_ylabel('Amplitude (µV)')
            for s,e in ictal_intervals:
                ax.axvspan(s, e, color='red', alpha=0.3)
            for s,e in preictal_intervals:
                ax.axvspan(s, e, color='yellow', alpha=0.3)

            if idx == len(channels_to_plot) - 1:
                ax.set_xlabel('Time (s)')

        plt.show()

        data = np.array(raw.get_data(picks=['eeg']))
        
        t = 1000
        plt.plot(np.arange(t), data[0][:t])
        plt.show()

        print(data.shape)
        

        print(raw.info.get_channel_types())
        print(len(raw.info.get_channel_types()))
        




