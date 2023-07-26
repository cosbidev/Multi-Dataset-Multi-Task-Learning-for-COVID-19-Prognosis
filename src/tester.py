
import torch

# Assuming you have a tensor of shape (32, 2) containing class scores
predictions = torch.randn(32, 2)

# Find the index of the maximum value along dimension 1
# (i.e., find the class with the highest score for each sample)
predicted_labels = torch.max(predictions, dim=1).indices

print(predicted_labels)

