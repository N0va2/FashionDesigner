import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn as nn
from transfer_models import FeatureExtractor, Classifier
import intel_extension_for_pytorch as ipex


# Assumptions: 
# 1. Using the same path as in test.py for loading the new dataset.
# 2. The new dataset has classification labels.
# 3. The new task is a classification task with 10 classes (change this as per your requirements).

# Hyperparameters
learning_rate = 0.0005
batch_size = 32
num_epochs = 25
num_classes = 10  # Change this based on your dataset

# Dataset loading and preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomResizedCrop(480),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Assuming RGB images
])

dataset = datasets.ImageFolder(root='Designer/ClothesFits', transform=transform)

# Splitting dataset into training and validation sets (80-20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the models, loss function, and optimizer
feature_extractor = FeatureExtractor().to('xpu')
classifier = Classifier(num_classes=num_classes).to('xpu')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to('xpu'), labels.to('xpu')
        
        # Forward pass
        features = feature_extractor(images)
        outputs = classifier(features)
        
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

print("Training complete!")
