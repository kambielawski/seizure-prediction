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

    d.plot_chb11()

    channel_selections = {
        2: ['P7-O1', 'P8-O2'],
        4: ['P7-O1', 'P8-O2', 'P3-O1', 'P4-O2'],
        6: ['P7-O1', 'P8-O2', 'P3-O1', 'P4-O2', 'T7-P7', 'T8-P8-0'],
        8: ['P7-O1', 'P8-O2', 'P3-O1', 'P4-O2', 'T7-P7', 'T8-P8-0', 'C3-P3', 'C4-P4'],
        9: ['P7-O1', 'P8-O2', 'P3-O1', 'P4-O2', 'T7-P7', 'T8-P8-0', 'C3-P3', 'C4-P4', 'CZ-PZ'],
    }

    # testing, training = d.random_5050split()
    # Regular train/test split using 4 patients for validation...    
    # Iterate over the channel selections
    for chs in channel_selections:
        n_tried = 0 
        results = {}
        # Perform 10-fold CV for each one 
        channels = channel_selections[chs]

        testing, training = d.train_test_split(['chb11', 'chb16', 'chb18', 'chb04'])
        # Balance dataset
        print('Getting balanced sample...')
        test_obs_balanced, test_labels_balanced = d.get_balanced_obs(testing, channels, interictal_multiplier=3)
        train_obs_balanced, train_labels_balanced = d.get_balanced_obs(training, channels, interictal_multiplier=3)

        print('Selecting channels...')
        # Select only the channels we care about 
        train_obs_selected = np.array(d.select_channels(train_obs_balanced, channels))
        test_obs_selected = np.array(d.select_channels(test_obs_balanced, channels))
        test_labels_balanced = np.array(test_labels_balanced)
        train_labels_balanced = np.array(train_labels_balanced)
        model_hist = m.train_simple_cnn(len(channels), train_obs_selected, train_labels_balanced, test_obs_selected, test_labels_balanced)

        with open(f'results/cnn_big2_unbalanced_results_{chs}.pkl', 'wb') as pf:
            pickle.dump(model_hist, pf)

        '''

        for loocv_patient, training in d.loocv_splits():
            # Balance dataset
            print('Getting balanced sample...')
            test_obs_balanced, test_labels_balanced = d.get_balanced_obs(loocv_patient, channels)
            train_obs_balanced, train_labels_balanced = d.get_balanced_obs(training, channels)

            print('Selecting channels...')
            # Select only the channels we care about 
            # train_obs_selected = np.array(d.select_channels(train_obs_balanced, channels)).transpose(0,2,1)
            # test_obs_selected = np.array(d.select_channels(test_obs_balanced, channels)).transpose(0,2,1)
            train_obs_selected = np.array(d.select_channels(train_obs_balanced, channels))
            test_obs_selected = np.array(d.select_channels(test_obs_balanced, channels))
            test_labels_balanced = np.array(test_labels_balanced)
            train_labels_balanced = np.array(train_labels_balanced)
            model_hist = m.train_simple_cnn(len(channels), train_obs_selected, train_labels_balanced, test_obs_selected, test_labels_balanced)

            patient_name = list(loocv_patient.keys())[0]
            results[patient_name] = model_hist

            with open(f'results/simplernn_results_{chs}ch_{patient_name}.pkl', 'wb') as pf:
                pickle.dump(results, pf)
            
            # After done with one LOOCV, break and move on. This is for trying a massive model 
            n_tried += 1
            if n_tried >= 3:
                break

        '''


    '''
    for loocv_patient, training in d.loocv_splits():
        # Balance dataset
        print('Getting balanced sample...')
        test_obs_balanced, test_labels_balanced = d.get_balanced_obs(loocv_patient, channels)
        train_obs_balanced, train_labels_balanced = d.get_balanced_obs(training, channels)

        print('Selecting channels...')
        # Select only the channels we care about 
        train_obs_selected = np.array(d.select_channels(train_obs_balanced, channels)).transpose(0,2,1)
        test_obs_selected = np.array(d.select_channels(test_obs_balanced, channels)).transpose(0,2,1)
        test_labels_balanced = np.array(test_labels_balanced)
        train_labels_balanced = np.array(train_labels_balanced)

        print('test')
        print(type(test_obs_selected))
        print(test_obs_selected.shape)
        print(test_obs_selected[0].shape)
        print(type(test_obs_selected[0]))
        print('train')
        print(type(train_obs_selected))
        print(train_obs_selected[0].shape)
        print(type(train_obs_selected[0]))
        print(train_obs_selected.shape)
        print(test_obs_selected.shape)

        model_hist = m.train_simple_cnn(len(channels), train_obs_selected, train_labels_balanced, test_obs_selected, test_labels_balanced)
    '''



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
    

    
