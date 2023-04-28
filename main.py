import mne 
import pickle
import numpy as np

from data_manager import DataManager
from model_manager import ModelManager

if __name__ == '__main__':
    d = DataManager()
    m = ModelManager()
    # d.load_chb_mit()
    # d.annotate_chbmit_dataset()

    # d.load_siena_scalp()

    d.load_data_from_pkl('data_pe600_epoch2.pkl')
    channels = ['P7-O1', 'P8-O2']

    print(d.data.keys())

    for loocv_patient, training in d.loocv_splits():
        # Balance dataset
        print('Balancing data...')
        test_obs_balanced, test_labels_balanced = d.get_balanced_obs(loocv_patient, channels)
        train_obs_balanced, train_labels_balanced = d.get_balanced_obs(training, channels)

        print('Selecting channels...')
        # Select only the channels we care about 
        train_obs_selected = d.select_channels(train_obs_balanced, channels)
        test_obs_selected = d.select_channels(test_obs_balanced, channels)

        print(np.array(train_obs_selected).shape)
        print(np.array(test_obs_selected).shape)

        m.train_simple_cnn(train_obs_selected, train_labels_balanced, test_obs_selected, test_labels_balanced)

    loocv_patient, training = d.get_loocv_split(patient=list(d.data.keys())[1])

    obs_test, labels_test = d.get_balanced_obs(loocv_patient)
    obs_train, labels_train = d.get_balanced_obs(training)

    with open('loocv_data_split.pkl', 'wb') as pf:
        pickle.dump(((obs_train, labels_train), (obs_test, labels_test)), pf)

    print([len(i) for i in [obs_test, labels_test]])
    print([len(i) for i in [obs_train, labels_train]])
    '''
    training_sample = d.get_balanced_sample(training) 
    single_patient = d.get_balanced_sample(loocv_patient)
    '''

    # print(d.count_epochs(d.data))

    '''
    patient_epochs = dict()
    print('\ttotal, ictal, preictal, interictal')
    for patient in list(d.data.keys()):
        loocv_patient, training = d.get_loocv_split(patient)
        patient_epochs[patient] = d.count_epochs(loocv_patient)
        print(patient, patient_epochs[patient])

    for patient in sorted(list(d.data.keys())):
        tot, ictal, pre, inter = patient_epochs[patient]
        print(f'{patient} & {inter} & {pre} & {ictal} & {tot} \\\\\n\hline')
    '''


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
    

    
