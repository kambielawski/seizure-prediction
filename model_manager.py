from data_manager import DataManager

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class EEGClassifier(nn.Module):
    def __init__(self):
        super(EEGClassifier, self).__init__()

        self.conv1 = nn.Sequential(
        nn.Conv1d(2, 16, kernel_size=2),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        self.conv2 = nn.Sequential(
        nn.Conv1d(16, 32, kernel_size=2),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(2)
        )

        self.fc1 = nn.Linear(32 * 256, 64)  # 256 is half of the input size (512) due to two max pooling layers
        self.fc2 = nn.Linear(64, 3)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class ModelManager:
    def __init__(self):
        pass

    def train_simple_cnn(self, train_obs, train_labels, test_obs, test_labels): 
        assert len(train_obs) == len(train_labels)
        assert len(test_obs) == len(test_labels)

        print(np.array(train_obs).shape)
        print(np.array(train_labels).shape)
        print(np.array(test_obs).shape)
        print(np.array(test_labels).shape)

        train_obs = torch.Tensor(train_obs)
        test_obs = torch.Tensor(test_obs)

        model = EEGClassifier()

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

        num_epochs = 50

        for epoch in range(num_epochs):
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
            print(f"Epoch: {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


        print('test loss: ', test_loss)
        print('test accuracy: ', test_accuracy)

        


