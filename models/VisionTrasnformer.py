import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import timm

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")       
else:
    device = torch.device("cpu")

print("Using device:", device)

train_dir = 'data/train'
test_dir = 'data/test'

print("Train:", train_dir)
print("Test:", test_dir)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)

model = timm.create_model(
    "vit_tiny_patch16_224",
    pretrained=True,
    num_classes=num_classes
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

def get_autocast():
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    elif device.type == "mps":
        return torch.amp.autocast("mps")   
    else:
        return torch.amp.autocast("cpu")

scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None


epochs = 15
train_losses = []
test_accuracies = []
all_true = []
all_pred = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")

    for images, labels in train_pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with get_autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        train_pbar.set_postfix(loss=loss.item())

    train_losses.append(running_loss / len(train_loader))

    model.eval()
    correct = 0
    total = 0

    test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Testing]")

    with torch.no_grad():
        for images, labels in test_pbar:
            images, labels = images.to(device), labels.to(device)

            with get_autocast():
                outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}: Loss={train_losses[-1]:.4f}, Test Acc={accuracy:.2f}%")

model.eval()
with torch.no_grad():
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    img = images[i].cpu().permute(1, 2, 0).numpy()
    img = img * 0.5 + 0.5
    axes[i].imshow(np.clip(img, 0, 1))

    pred_class = train_dataset.classes[preds[i]]
    true_class = train_dataset.classes[labels[i]]

    color = "green" if pred_class == true_class else "red"
    axes[i].set_title(f"Pred: {pred_class}\nTrue: {true_class}", color=color)
    axes[i].axis("off")

plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, marker='s', color='red')
plt.title("Test Accuracy")
plt.xlabel("Epoch")

plt.show()


cm = confusion_matrix(all_true, all_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(18, 15))
sns.heatmap(cm_norm, cmap='Blues',
            xticklabels=train_dataset.classes,
            yticklabels=train_dataset.classes)
plt.title("Confusion Matrix (ViT-Tiny MPS)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


print("\n=== Classification Report (ViT-Tiny, MPS) ===")
print(classification_report(
    all_true, all_pred, target_names=train_dataset.classes
))



