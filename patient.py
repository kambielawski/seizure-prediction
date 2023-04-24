import numpy as np
import mne

class Patient:
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset
        self.age = None
        self.gender = None
        self.seizure = None
        self.localization = None
        self.lateralization = None
        self.num_channels = None
        self.num_seizures = None
        self.total_rec_time = None
        self.rec_times = dict()
        self.edf_files = dict() 
        self.extra_files = []
        self.seizures = []

    def add_edf_file(self, file_name):
        self.edf_files[file_name] = dict()

    def add_extra_file(self, file_name):
        self.extra_files.append(file_name)

    def add_seizure(self, file, seizure):
        self.edf_files[file]['seizures'].append(seizure)

    def add_channel_map(self, file, channel_map):
        assert channel_map != None
        self.edf_files[file]['channel_map'] = channel_map

    def add_rec_frequency(self, file, rec_frequency):
        self.edf_files[file]['rec_frequency'] = rec_frequency
    
    def add_rec_time(self, file, rec_start, rec_end, rec_length):
        if file not in self.edf_files:
            self.edf_files[file] = dict()

        self.edf_files[file]['rec_start'] = rec_start
        self.edf_files[file]['rec_end'] = rec_end
        self.edf_files[file]['rec_length'] = rec_length
        self.rec_times[file] = rec_length

    def set_age(self, age):
        self.age = age
    def set_gender(self, gender):
        self.gender = gender
    def set_seizure(self, seizure):
        self.seizure = seizure
    def set_localization(self, localization):
        self.localization = localization
    def set_lateralization(self, lateralization):
        self.lateralization = lateralization
    def set_num_channels(self, num_channels):
        self.num_channels = num_channels
    def set_num_seizures(self, num_seizures):
        self.num_seizures = num_seizures
    def set_total_rec_time(self, total_rec_time):
        self.total_rec_time = total_rec_time

    def get_total_rec_time(self):
        print('csv:',self.total_rec_time)
        print('calc:',sum([self.rec_times[t].seconds / 60 for t in self.rec_times]))
        return self.total_rec_time


