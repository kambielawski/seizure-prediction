import pickle
import os

last_epoch_accuracies = [] 
for file in os.listdir('./results/simple_cnn'):
    with open(f'./results/simple_cnn/{file}', 'rb') as pf:
        res = pickle.load(pf)
        patient = file[-9:-4]
        last_epoch_accuracy = float(res[patient]['val_accuracy_history'][-1])
        last_epoch_accuracies.append(last_epoch_accuracy)
print(last_epoch_accuracies)
