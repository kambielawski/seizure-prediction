from data_manager import DataManager

import numpy as np
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

class CNN_BigK(nn.Module):
    def __init__(self, n_channels = 2):
        super(EEGClassifier, self).__init__()

        k_size = 25
        padding = 12

        self.conv1 = nn.Sequential(
        nn.Conv1d(n_channels, 32, kernel_size=25, padding=padding), # n channels, 32 kernels, k_size=24
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
        # print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


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
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())
        x = self.conv3(x)
        # print(x.size())
        x = self.conv4(x)
        # print(x.size())
        x = x.view(x.size(0), -1)  # Flatten the tensor
        # print(x.size())
        x = self.dropout(F.relu(self.fc1(x)))
        # print(x.size())
        x = self.fc2(x)
        # print(x.size())
        return x


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
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())
        x = self.conv3(x)
        # print(x.size())
        x = self.conv4(x)
        # print(x.size())
        x = x.view(x.size(0), -1)  # Flatten the tensor
        # print(x.size())
        x = self.dropout(F.relu(self.fc1(x)))
        # print(x.size())
        x = self.fc2(x)
        # print(x.size())
        return x

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
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # print(x.size())

        # Forward propagate the RNN
        out, _ = self.rnn(x, h0)

        # Pass the output of the last time step to the fully connected layer
        out = self.fc(out[:, -1, :])

        return out


class RNN_LSTM(nn.Module):
    def __init__(self, n_channels=2, hidden_size=64, num_layers=2, num_classes=3):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(n_channels, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class ModelManager:
    def __init__(self):
        pass

    def train_simple_cnn(self, n_channels, train_obs, train_labels, test_obs, test_labels): 
        assert len(train_obs) == len(train_labels)
        assert len(test_obs) == len(test_labels)

        print(np.array(train_obs).shape)
        print(np.array(train_labels).shape)
        print(np.array(test_obs).shape)
        print(np.array(test_labels).shape)

        train_obs = torch.Tensor(train_obs)
        test_obs = torch.Tensor(test_obs)
        # train_obs = train_obs.permute(0,2,1)
        # test_obs = test_obs.permute(0,2,1)

        # model = SimpleCNN_BigK(n_channels)
        device = torch.device("mps")
        model = CNN_4layer_big(n_channels).to(device)

        # Create DataLoaders for training and validation sets
        train_data = TensorDataset(train_obs, torch.LongTensor(train_labels))
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        val_data = TensorDataset(test_obs, torch.LongTensor(test_labels))
        val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

        # print('train data shape:')
        # print(train_data.tensors[0].shape) 


        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 20

        val_accuracy_history = []
        train_accuracy_history = []


        for epoch in range(num_epochs):
            start = time.perf_counter()
            # Training loop
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
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}")

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
            print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}")
            end = time.perf_counter()
            epoch_time = end - start
            print(f'Epoch time: {epoch_time}')


        '''

        for epoch in range(num_epochs):
            # Train model
            model.train()
            for inputs, labels in train_loader:
                inputs = inputs.permute(0, 2, 1)  # Swap the dimensions to match the expected input shape (batch_size, num_channels, length)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.permute(0, 2, 1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_accuracy += torch.sum(preds == labels.data)

            val_loss /= len(val_loader)
            val_accuracy = val_accuracy / len(val_data)

            val_accuracy_history.append(val_accuracy)
            print(f"Epoch: {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
'''

        return { 'model_state_dict': model.state_dict(), 'val_accuracy_history': val_accuracy_history, 'train_accuracy_history': train_accuracy_history }



    def load_model_from_pkl(self, pkl_file):
        with open(pkl_file, 'rb') as pf:
            res = pickle.load(pf)
        return res['model_state_dict']


