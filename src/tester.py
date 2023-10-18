import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Sample dataset
class CustomDataset(Dataset):
    def __init__(self, num_samples_per_class, class_balance=(0.5, 0.5)):
        self.data = []
        self.labels = []
        for i in range(2):
            self.data.extend(torch.randn(num_samples_per_class, 2) + i * 2)  # Sample data points
            self.labels.extend([i] * num_samples_per_class)
        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)

        num_samples = len(self.data)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # Adjust the data according to the desired class balance
        num_samples_class_0 = int(class_balance[0] * num_samples / 2)
        num_samples_class_1 = num_samples_per_class - num_samples_class_0
        final_indices = np.concatenate((indices[:num_samples_class_0], indices[num_samples_per_class:num_samples_per_class + num_samples_class_1]))

        self.data = self.data[final_indices]
        self.labels = self.labels[final_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)


# Hyperparameters
learning_rate = 0.001
num_epochs = 10
num_samples_per_class = 1000
batch_size = 32
balances = [(i, 1 - i) for i in np.linspace(0, 1, num_epochs)]

# Model, loss and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
for epoch in range(num_epochs):
    dataset = CustomDataset(num_samples_per_class, balances[epoch])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, (data, labels) in enumerate(dataloader):
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Class Balance: {balances[epoch]}")


