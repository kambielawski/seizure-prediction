import pickle
import numpy as np

from data_manager import DataManager
from model_manager import ModelManager

if __name__ == '__main__':
    d = DataManager()
    m = ModelManager()

    ######## Uncomment to load and annotate data ##########
    #### Data will be stored in data_pe600_epoch2.pkl #####

    # d.load_chb_mit()
    # d.annotate_chbmit_dataset()

    #######################################################

    d.load_data_from_pkl('data_pe600_epoch2.pkl')

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
        test_obs_balanced, test_labels_balanced = d.get_balanced_obs(testing, channels)
        train_obs_balanced, train_labels_balanced = d.get_balanced_obs(training, channels, interictal_multiplier=3)

        print('Selecting channels...')
        # Select only the channels we care about 
        train_obs_selected = np.array(d.select_channels(train_obs_balanced, channels))
        test_obs_selected = np.array(d.select_channels(test_obs_balanced, channels))
        test_labels_balanced = np.array(test_labels_balanced)
        train_labels_balanced = np.array(train_labels_balanced)
        model_hist = m.train_model(len(channels), train_obs_selected, train_labels_balanced, test_obs_selected, test_labels_balanced)

        m.save_model_to_file(model_hist, f'results/cnn_big2_unbalanced_results_{chs}.pkl')

        ##### Uncomment to use Leave-One-Out Cross-Validation ######
        '''
        for loocv_patient, training in d.loocv_splits():
            # Balance dataset
            print('Getting balanced sample...')
            test_obs_balanced, test_labels_balanced = d.get_balanced_obs(loocv_patient, channels)
            train_obs_balanced, train_labels_balanced = d.get_balanced_obs(training, channels)

            print('Selecting channels...')
            # Select only the channels we care about 
            train_obs_selected = np.array(d.select_channels(train_obs_balanced, channels))
            test_obs_selected = np.array(d.select_channels(test_obs_balanced, channels))
            test_labels_balanced = np.array(test_labels_balanced)
            train_labels_balanced = np.array(train_labels_balanced)
            model_hist = m.train_simple_cnn(len(channels), train_obs_selected, train_labels_balanced, test_obs_selected, test_labels_balanced)

            patient_name = list(loocv_patient.keys())[0]
            results[patient_name] = model_hist

            with open(f'results/simplernn_results_{chs}ch_{patient_name}.pkl', 'wb') as pf:
                pickle.dump(results, pf)
        '''
    

    
