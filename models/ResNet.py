import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  
import os
import shutil
import random

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns  


# 1. Define dataset paths
data_dir = 'PlantVillage'  
train_dir = './data/train'  
test_dir = './data/test'  

# 2. Create train/test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 3. Loop through each class folder
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_dir):
        continue  

    # Create class folders inside train/test
    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Get all images in this class
    images = [img for img in os.listdir(class_dir) if img.endswith(('.jpg', '.png', '.jpeg', '.JPG'))]

    # Check image count
    if len(images) == 0:
        print(f"Class {class_name}: No images found, skipping.")
        continue

    # Shuffle images
    random.shuffle(images)

    # 80/20 split
    split_index = int(len(images) * 0.8)  
    train_images = images[:split_index]  
    test_images = images[split_index:]  

    # Copy training images
    for img in train_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(train_class_dir, img)
        shutil.copy(src, dst)

    # Copy test images
    for img in test_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(test_class_dir, img)
        shutil.copy(src, dst)

    print(f"Class {class_name}: {len(train_images)} train images, {len(test_images)} test images")

print("Dataset split completed!")


# Transforms and Dataloaders
transform = transforms.Compose([
    transforms.Resize((128, 128)),          
    transforms.ToTensor(),                   
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))     
])
train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
test_dataset = datasets.ImageFolder(root='./data/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ResNet building blocks
class BasicBlock(nn.Module):
    expansion = 1  

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels,
                            stride, downsample))
        self.in_channels = out_channels * block.expansion  
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv → BN → ReLU → MaxPool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pool → flatten → FC
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# Training setup
num_classes = len(train_dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2],
               num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(model)

epochs = 15
train_losses = []  
test_accuracies = []  


# Training loop
for epoch in range(epochs):
    model.train()  
    running_loss = 0.0

    train_progress = tqdm(train_loader,
                          desc=f"Epoch {epoch + 1}/{epochs} [Training]")
    for images, labels in train_progress:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  
        outputs = model(images)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  

        running_loss += loss.item()
        train_progress.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    # Evaluation on test set (for accuracy only here)
    model.eval()  
    correct = 0
    total = 0

    test_progress = tqdm(test_loader,
                         desc=f"Epoch {epoch + 1}/{epochs} [Testing]")
    with torch.no_grad():
        for images, labels in test_progress:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_accuracy = 100 * correct / total
    test_accuracies.append(epoch_accuracy)

    print(f"Epoch [{epoch + 1}/{epochs}], "
          f"Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_accuracy:.2f}%")


# Sample predictions
model.eval()
with torch.no_grad():
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    img = images[i].cpu().permute(1, 2, 0).numpy()  
    img = img * 0.5 + 0.5  
    axes[i].imshow(np.clip(img, 0, 1))

    pred_class = train_dataset.classes[predicted[i]]
    true_class = train_dataset.classes[labels[i]]

    axes[i].set_title(
        f"Pred: {pred_class}\nTrue: {true_class}",
        color="green" if pred_class == true_class else "red"
    )
    axes[i].axis('off')

plt.show()


# Loss and accuracy curves
plt.figure(figsize=(12, 5))
# Loss curve
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, marker='o', color='blue')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(alpha=0.3)

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), test_accuracies, marker='s', color='red')
plt.title('Test Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# Final evaluation: confusion matrix + classification report
model.eval()
all_true_labels = []
all_pred_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        all_true_labels.extend(labels.cpu().numpy())
        all_pred_labels.extend(predicted.cpu().numpy())

print("\n=== Confusion Matrix Visualization (ResNet) ===")
cm = confusion_matrix(all_true_labels, all_pred_labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(18, 15))
sns.heatmap(
    cm_normalized,
    annot=False,
    cmap='Blues',
    xticklabels=train_dataset.classes,
    yticklabels=train_dataset.classes
)
plt.title('Normalized Confusion Matrix (Plant Disease ResNet)', fontsize=16)
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('True Class', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("\n=== Classification Report (ResNet) ===")
print(classification_report(
    all_true_labels,
    all_pred_labels,
    target_names=train_dataset.classes
))

print(f"\nMatrix shape: {cm.shape} (num classes: {num_classes})")
print("First 5 classes prediction distribution:")
for i in range(min(5, num_classes)):
    true_class = train_dataset.classes[i]
    top_pred_idx = cm[i].argmax()
    top_pred_class = train_dataset.classes[top_pred_idx]
    print(f"  {true_class} → Most predicted: {top_pred_class} (count: {cm[i][top_pred_idx]})")

