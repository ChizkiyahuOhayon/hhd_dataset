import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from custom_dataset import HHD_dataset


testing_label = 'TEST/testing_labels'
training_label = 'TRAIN/training_labels'
training_path = 'TRAIN/Train'
testing_path = 'TEST/Test'

my_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([100, 40]),
    transforms.ToTensor()
])

training_set = HHD_dataset(image_dir=training_path, label_csv=training_label, transform=my_transform)
testing_set = HHD_dataset(image_dir=testing_path, label_csv=testing_label, transform=my_transform)

train_loader = DataLoader(dataset=training_set, shuffle=True, batch_size=4)
test_loader = DataLoader(dataset=testing_set, shuffle=True, batch_size=4)

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=27):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=30, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(30)
        self.conv3 = nn.Conv2d(in_channels=30, out_channels=60, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(60)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(60, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = torch.squeeze(x)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels = 1
num_classes = 27
learning_rate = 3e-4 # karpathy's constant
batch_size = 4
num_epochs = 50



# Initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    print(f"epoch {epoch}")
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")




