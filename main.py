from data_manager import DataManager
import mne 
import numpy as np

if __name__ == '__main__':
    d = DataManager()
    # d.load_chb_mit()
    # d.annotate_chbmit_dataset()

    # d.load_siena_scalp()

    d.load_data_from_pkl('data_pe600_epoch2.pkl')
    d.count_epochs()

    # patient = d.patients[0]

    '''
    for patient in d.patients:
        for edf_file in patient.edf_files:
            print(patient.edf_files[edf_file])
            channels = patient.edf_files[edf_file]['channel_map']
            for channel in channels:
                print(f'{channel}: ' + channels[channel])
            input()

    key = list(patient.edf_files.keys())[0]
    print(key)
    file_name = patient.edf_files[key]['file_path']
    print(file_name)

    d.load_edf(file_name, patient)
    '''
    
    # print(patient.edf_files)
    # edf_file = patient.edf_files[0]

    # d.load_edf(edf_file, patient)
    # print(len(d.seizures))
    

    
