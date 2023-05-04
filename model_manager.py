from data_manager import DataManager

import numpy as np
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

'''
"Big K" model
'''
class SimpleCNN_BigK(nn.Module):
    def __init__(self, n_channels = 2):
        super(SimpleCNN_BigK, self).__init__()

        k_size = 25
        padding = 12

        self.conv1 = nn.Sequential(
        nn.Conv1d(n_channels, 16, kernel_size=k_size, padding=padding), # 2 channels, 16 kernels, k_size=2
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        self.conv2 = nn.Sequential(
        nn.Conv1d(16, 32, kernel_size=13, padding=6),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        # self.fc1 = nn.Linear(32 * 256, 64)  # 256 is half of the input size (512) due to two max pooling layers
        self.fc1 = nn.Linear(16 * 256, 32)
        # self.fc2 = nn.Linear(64, 3)
        self.fc2 = nn.Linear(32, 3)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

'''
Kernel size 2 on a Conv1D? What was I thinking?! 
This is the model I attempted at the very beginning 
'''
class SimpleCNN(nn.Module):
    def __init__(self, n_channels = 2):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Sequential(
        nn.Conv1d(n_channels, 16, kernel_size=2, padding=1), # 2 channels, 16 kernels, k_size=2
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        self.conv2 = nn.Sequential(
        nn.Conv1d(16, 32, kernel_size=2, padding=1),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        # self.fc1 = nn.Linear(32 * 256, 64)  # 256 is half of the input size (512) due to two max pooling layers
        self.fc1 = nn.Linear(16 * 256, 32)
        # self.fc2 = nn.Linear(64, 3)
        self.fc2 = nn.Linear(32, 3)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

'''
A bit bigger model 
'''
class CNN_2x(nn.Module):
    def __init__(self, n_channels = 2):
        super(EEGClassifier, self).__init__()

        self.conv1 = nn.Sequential(
        nn.Conv1d(n_channels, 32, kernel_size=n_channels, padding=n_channels//2), # 2 channels, 16 kernels, k_size=2
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        self.conv2 = nn.Sequential(
        nn.Conv1d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        # self.fc1 = nn.Linear(32 * 256, 64)  # 256 is half of the input size (512) due to two max pooling layers
        self.fc1 = nn.Linear(32 * 256, 64)
        # self.fc2 = nn.Linear(64, 3)
        self.fc2 = nn.Linear(64, 3)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

'''
This is when I started to wise up to larger kernel sizes
'''
class CNN_HugeK(nn.Module):
    def __init__(self, n_channels = 2):
        super(CNN_HugeK, self).__init__()

        # k_size = 501
        # padding = 250
        k_size = 401
        padding = 200

        self.conv1 = nn.Sequential(
        nn.Conv1d(n_channels, 32, kernel_size=k_size, padding=padding), # n channels, 32 kernels, k_size=24
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        self.conv2 = nn.Sequential(
        nn.Conv1d(32, 64, kernel_size=13, padding=6),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        # self.fc1 = nn.Linear(32 * 256, 64)  # 256 is half of the input size (512) due to two max pooling layers
        self.fc1 = nn.Linear(32 * 256, 64)
        # self.fc2 = nn.Linear(64, 3)
        self.fc2 = nn.Linear(64, 3)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))
        # print(x.size())
        x = self.fc2(x)
        return x

'''
Started to add layers... 
'''
class CNN_3layer(nn.Module):
    def __init__(self, n_channels = 2):
        super(CNN_3layer, self).__init__()

        k_size = 401
        padding = 200

        self.conv1 = nn.Sequential(
        nn.Conv1d(n_channels, 32, kernel_size=k_size, padding=padding), # n channels, 32 kernels, k_size=24
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        k_size2 = 151
        padding2 = 75

        self.conv2 = nn.Sequential(
        nn.Conv1d(32, 64, kernel_size=k_size2, padding=padding2),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        k_size3 = 15
        padding3 = 7

        self.conv3 = nn.Sequential(
        nn.Conv1d(64, 64, kernel_size=k_size3, padding=padding3),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        # self.fc1 = nn.Linear(32 * 256, 64)  # 256 is half of the input size (512) due to two max pooling layers
        self.fc1 = nn.Linear(32 * 128, 64)
        # self.fc2 = nn.Linear(64, 3)
        self.fc2 = nn.Linear(64, 3)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

'''
Much larger, 4 layer model. 
'''
class CNN_4layer(nn.Module):
    def __init__(self, n_channels = 2):
        super(CNN_4layer, self).__init__()

        k_size = 401
        padding = 200

        self.conv1 = nn.Sequential(
        nn.Conv1d(n_channels, 32, kernel_size=k_size, padding=padding), # n channels, 32 kernels, k_size=24
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(4)
        )

        k_size2 = 151
        padding2 = 75

        self.conv2 = nn.Sequential(
        nn.Conv1d(32, 64, kernel_size=k_size2, padding=padding2),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        k_size3 = 55
        padding3 = 27

        self.conv3 = nn.Sequential(
        nn.Conv1d(64, 128, kernel_size=k_size3, padding=padding3),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        k_size4 = 15
        padding4 = 7

        self.conv4 = nn.Sequential(
        nn.Conv1d(128, 128, kernel_size=k_size4, padding=padding4),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        # self.fc1 = nn.Linear(32 * 256, 64)  # 256 is half of the input size (512) due to two max pooling layers
        self.fc1 = nn.Linear(32 * 64, 128)
        # self.fc2 = nn.Linear(64, 3)
        self.fc2 = nn.Linear(128, 3)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


'''
This is the model I finally landed on for deployment testing
'''
class CNN_4layer_big(nn.Module):
    def __init__(self, n_channels = 2):
        super(CNN_4layer_big, self).__init__()

        k_size = 451
        padding = 225

        self.conv1 = nn.Sequential(
        # nn.Conv1d(n_channels, 128, kernel_size=k_size, padding=padding), # n channels, 32 kernels, k_size=24
        nn.Conv1d(n_channels, 256, kernel_size=k_size, padding=padding), # n channels, 32 kernels, k_size=24
        # nn.BatchNorm1d(128),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.MaxPool1d(4)
        )

        # k_size2 = 101
        # padding2 = 50
        k_size2 = 201
        padding2 = 100

        self.conv2 = nn.Sequential(
        nn.Conv1d(256, 128, kernel_size=k_size2, padding=padding2),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        # k_size3 = 31
        # padding3 = 15
        k_size3 = 101
        padding3 = 50

        self.conv3 = nn.Sequential(
        # nn.Conv1d(64, 64, kernel_size=k_size3, padding=padding3),
        nn.Conv1d(128, 64, kernel_size=k_size3, padding=padding3),
        # nn.BatchNorm1d(64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        # k_size4 = 15
        # padding4 = 7
        k_size4 = 51
        padding4 = 25

        self.conv4 = nn.Sequential(
        # nn.Conv1d(64, 32, kernel_size=k_size4, padding=padding4),
        nn.Conv1d(64, 32, kernel_size=k_size4, padding=padding4),
        # nn.BatchNorm1d(32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        # self.fc1 = nn.Linear(32 * 256, 64)  # 256 is half of the input size (512) due to two max pooling layers
        self.fc1 = nn.Linear(32 * 16, 128)
        # self.fc2 = nn.Linear(64, 3)
        self.fc2 = nn.Linear(128, 3)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
'''
A GRU model 
'''
class SimpleRNN(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, num_layers=4, num_classes=3):
        super(SimpleRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Allocate memory for M1
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0) # forward
        out = self.fc(out[:, -1, :])

        return out

'''
An LSTM model
'''
class RNN_LSTM(nn.Module):
    def __init__(self, n_channels=2, hidden_size=64, num_layers=2, num_classes=3):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(n_channels, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # allocate mem for M1
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # forward
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])
        return out


class ModelManager:
    def __init__(self):
        pass

    '''
    Main entry point to training a model
    '''
    def train_model(self, n_channels, train_obs, train_labels, test_obs, test_labels): 
        assert len(train_obs) == len(train_labels)
        assert len(test_obs) == len(test_labels)

        print('Training set size: ', np.array(train_obs).shape[0])
        print('Test set size: ', np.array(test_obs).shape[0])

        train_obs = torch.Tensor(train_obs)
        test_obs = torch.Tensor(test_obs)

        # This may be necessary for RNN models:

        # train_obs = train_obs.permute(0,2,1)
        # test_obs = test_obs.permute(0,2,1)

        device = torch.device("mps") # Only works on M1 Mac 
        model = CNN_4layer_big(n_channels).to(device)

        # Create DataLoaders for training and validation sets
        train_data = TensorDataset(train_obs, torch.LongTensor(train_labels))
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        val_data = TensorDataset(test_obs, torch.LongTensor(test_labels))
        val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 20

        val_accuracy_history = []
        train_accuracy_history = []

        # Training loop
        for epoch in range(num_epochs):
            start_time = time.perf_counter()
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_acc = 100 * correct / total
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")

            # Validation loop
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            val_accuracy = 100 * correct / total

            val_accuracy_history.append(val_accuracy)
            train_accuracy_history.append(train_acc)
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}")
            end_time = time.perf_counter()
            epoch_time = end_time - start_time
            print(f'Epoch time: {epoch_time}')

        return { 'model_state_dict': model.state_dict(), 'val_accuracy_history': val_accuracy_history, 'train_accuracy_history': train_accuracy_history }

    '''
    Load a model's weights/biases from a pkl'd file
    '''
    def load_model_from_pkl(self, pkl_file):
        with open(pkl_file, 'rb') as pf:
            res = pickle.load(pf)
        return res['model_state_dict']
    
    '''
    Save a model's weights/biases and train/validation history to file
    '''
    def save_model_to_file(self, model_hist, file_name):
        with open(file_name, 'wb') as pf:
            pickle.dump(model_hist, pf)


