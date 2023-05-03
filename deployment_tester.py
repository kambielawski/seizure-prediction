import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from data_manager import DataManager
from model_manager import ModelManager, CNN_4layer_big

parser = argparse.ArgumentParser()
parser.add_argument('results_file')
args = parser.parse_args()

channel_selections = {
    # 2: ['P7-O1', 'P8-O2'],
    # 4: ['P7-O1', 'P8-O2', 'P3-O1', 'P4-O2'],
    # 6: ['P7-O1', 'P8-O2', 'P3-O1', 'P4-O2', 'T7-P7', 'T8-P8-0'],
    # 8: ['P7-O1', 'P8-O2', 'P3-O1', 'P4-O2', 'T7-P7', 'T8-P8-0', 'C3-P3', 'C4-P4'],
    9: ['P7-O1', 'P8-O2', 'P3-O1', 'P4-O2', 'T7-P7', 'T8-P8-0', 'C3-P3', 'C4-P4', 'CZ-PZ'],
}

device = torch.device("mps")

d = DataManager()
m = ModelManager()

def visualize_deployment(edf_dict, preds, file, n_channels):
    ictal_intervals = edf_dict['ictal_intervals']
    preictal_intervals = edf_dict['preictal_intervals']

    plt.scatter(range(len(preds)), preds)
    plt.title(f'{file} Predictions ({n_channels} channels)')
    plt.yticks([0,1,2], ['Interictal', 'Preictal', 'Ictal'])
    plt.xlabel('Epochs')
    plt.ylabel('Classification')
    for s,e in ictal_intervals:
        plt.axvspan(s//2, e//2, color='red', alpha=0.3, label="True Ictal")
    for s,e in preictal_intervals:
        plt.axvspan(s//2, e//2, color='yellow', alpha=0.3, label="True Preictal")
    plt.legend()
    plt.show()

def visualize_with_warnings(edf_dict, warnings, preds, file, n_channels):
    ictal_intervals = edf_dict['ictal_intervals']
    preictal_intervals = edf_dict['preictal_intervals']

    plt.scatter(range(len(preds)), preds)
    plt.title(f'{file} Predictions ({n_channels} channels)')
    plt.yticks([0,1,2], ['Interictal', 'Preictal', 'Ictal'])
    plt.xlabel('Epochs')
    plt.ylabel('Classification')
    for i, warn in enumerate(warnings):
        if warn == 1:
            plt.axvline(i, color='blue')
            
    for s,e in ictal_intervals:
        plt.axvspan(s//2, e//2, color='red', alpha=0.3, label="True Ictal")
    for s,e in preictal_intervals:
        plt.axvspan(s//2, e//2, color='yellow', alpha=0.3, label="True Preictal")
    plt.legend()
    plt.show()

def test_deployment(model, edf_dict, n_channels):
    epochs = np.array([np.array(e) for e in edf_dict['epochs'].get_data(picks=channel_selections[n_channels])])
    labels = edf_dict['labels']
    ictal_intervals = edf_dict['ictal_intervals']
    preictal_intervals = edf_dict['preictal_intervals']
    n_epochs = len(epochs) 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    val_data = TensorDataset(torch.Tensor(epochs), torch.LongTensor(labels))
    val_loader = DataLoader(val_data, batch_size=n_epochs, shuffle=False) # Shuffle is false to mimic continuous monitoring

    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for inputs, outputs in val_loader:
            # inputs, outputs = inputs.to(device), outputs.to(device)
            # epochs = inputs.permute(0, 2, 1)

            outputs = model(inputs)
            loss = criterion(outputs, torch.LongTensor(labels))

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
    
    return np.array(preds)


def deployment_warnings(preds, true, k=3):
    warnings = []
    n_warnings = 0
    true_warnings = 0
    n_epochs = len(true)
    for i, c in enumerate(true):
        # If we predict k times in a row, issue warning 
        if i > k and all([z > 0 for z in preds[i-k:i]]) and warnings[-1] == 0: 
            print(preds[i-k:i])
            warnings.append(1)
            n_warnings += 1
            if c > 0:
                true_warnings += 1
        else:
            warnings.append(0)
    fprate_per_hour = (n_warnings - true_warnings) / (n_epochs*2 / 3600)
    if n_warnings > 0:
        print(f'warning rate: {true_warnings}/{n_warnings}=', true_warnings/n_warnings)
    print(f'fprate/hour: {fprate_per_hour}')
    return warnings, (n_warnings, true_warnings)


# Best k is considered to be a balance between high seizure warning accuracy
# and a low false positive rate.
def find_best_k(model, testing, n_channels):
    global d
    k2fprates = {}
    k2accuracies = {}
    for k in range(2,6):
        valid_files = []
        total_warnings = 0 
        total_true_warnings = 0
        total_seizures = 0
        total_seizures_warned = 0
        for patient in testing:
            for file in testing[patient]:
                # Ensure all channels are actually present in the data... 
                if len(testing[patient][file]['ictal_intervals']) > 0:
                    if all([ch in testing[patient][file]['epochs'][0].ch_names for ch in channel_selections[n_channels]]):
                        valid_files.append(file) 
                        # Predictions vs true values 
                        preds = test_deployment(model, testing[patient][file], n_channels)
                        true = testing[patient][file]['labels']
                        # Get warnings
                        warnings, (n_warnings, true_warnings) = deployment_warnings(preds, true, k)

                        # visualize_with_warnings(testing[patient][file], warnings, preds, file, n_channels)

                        # Get accuracy measures
                        n_seizures = len(testing[patient][file]['ictal_intervals'])
                        seizures_warned = 0
                        for s,e in testing[patient][file]['ictal_intervals']:
                            if sum(warnings[s//2:e//2]) > 0:
                                seizures_warned += 1

                        print('seizures: ', n_seizures, ', seizures warned: ', seizures_warned)
                        total_warnings += n_warnings
                        total_true_warnings += true_warnings
                        total_seizures += n_seizures
                        total_seizures_warned += seizures_warned

        total_epochs, _, _, _ = d.count_epochs(testing, valid_files)
        print(total_epochs, (total_epochs*2/3600), total_warnings, total_true_warnings)
        k2fprates[k] = (total_warnings - total_true_warnings) / (total_epochs*2 / 3600)
        k2accuracies[k] = total_seizures_warned / total_seizures
        print(k2fprates, k2accuracies)

    return k2fprates, k2accuracies


if __name__ == '__main__':

    n_channels = int(args.results_file[-5])
    
    # Load data
    d.load_data_from_pkl('data_pe600_epoch2.pkl')

    # Load in model and weights/biases

    testing, training = d.train_test_split(['chb11', 'chb16', 'chb18', 'chb04'])

    nch2fprate = {}
    nch2accuracies = {}
    for nch in channel_selections:
        model = CNN_4layer_big(nch) 
        model_file = args.results_file.replace('2.', str(nch)+'.')
        print(model_file)
        model_state_dict = m.load_model_from_pkl(model_file)
        model.load_state_dict(model_state_dict)
        fprates, accuracies = find_best_k(model, testing, nch)
        nch2fprate[nch] = fprates 
        nch2accuracies[nch] = accuracies
        print(fprates, accuracies)
    with open('fprates.pkl', 'wb') as pf:
        pickle.dump(nch2fprate, pf)
    with open('accuracies.pkl', 'wb') as pf:
        pickle.dump(nch2accuracies, pf)
    exit(1)

    print(n_channels)

    model = CNN_4layer_big(n_channels) 
    model_file = args.results_file
    model_state_dict = m.load_model_from_pkl(model_file)
    model.load_state_dict(model_state_dict)
    # model_file = args.results_file.replace('2.', str(nch)+'.')
    p = 'chb11'
    for file in testing[p]:
        if len(testing[p][file]['ictal_intervals']) > 0:
            preds = test_deployment(model, testing[p][file], n_channels)
            warnings, _ = deployment_warnings(preds, testing[p][file]['labels'])
            # visualize_with_warnings(testing[p][file], warnings, preds, file, n_channels)
            visualize_deployment(testing[p][file], preds, file, n_channels)

    exit(1)

