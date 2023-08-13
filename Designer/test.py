import torch
import torchvision.models as models
import intel_extension_for_pytorch as ipex
import torch.nn as nn
import os

MODEL_DIR = './models2'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print(f"Device count: {torch.xpu.device_count()}")
for i in range(torch.xpu.device_count()):
    print(f"Device {i}: {torch.xpu.get_device_properties(i)}")

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)
dtype=torch.float32

model = model.to('xpu')
data = data.to('xpu')

model = ipex.optimize(model, dtype=dtype)

########## FP32 ############
with torch.no_grad():
    #model = torch.jit.trace(model, data)
    #model = torch.jit.freeze(model)

    model(data)

import torch

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))


import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

preprocess = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize images to 224x224. You can adjust this as per your model's input size
    transforms.ToTensor(),  # Convert image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 5])  # Normalize. Values are standard for pre-trained models. Adjust if needed.
])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    preprocess
])

dataset_path = 'Designer/ClothesFits'

dataset = ImageFolder(root = dataset_path, transform=preprocess)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = preprocess

from torch.utils.data import DataLoader

batch_size = 16
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    plt.ion()
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(10,10))  # Adjust as necessary
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.pause(1)
    plt.close()

import torchvision.utils

# Get a batch of images and display them
images, labels = next(iter(dataloader))
imshow(torchvision.utils.make_grid(images))

from PIL import Image

def save_sample(epoch, fixed_input, generator):
    with torch.no_grad():
        generated = generator(fixed_input).detach().cpu()
    grid = torchvision.utils.make_grid(generated, nrow=6, padding=2, normalize=True)  # Adjusted nrow to 2
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    image = Image.fromarray(ndarr)
    image.save(os.path.join('./generated_images_3', f"sample_epoch_{epoch+813}.jpg"))
    imshow(grid)


from models_old import Generator, Discriminator

z_dim = 100
fixed_input = torch.randn(64, z_dim, 1, 1).to('xpu')

num_epochs = 1000

# Hyperparameters
lr = 0.0005
betas = (0.5, 0.999)

# Define the initialize_weights function
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

# Create instances of Generator and Discriminator
#old_generator = Generator(z_dim).to('xpu')  # Assuming you're generating 224x224 RGB images
#old_discriminator = Discriminator().to('xpu')

from models import NewGenerator, NewDiscriminator

new_generator = NewGenerator(z_dim).to('xpu')
new_discriminator = NewDiscriminator().to('xpu')

#initialize_weights(new_generator)
#initialize_weights(new_discriminator)

#for (name_pre, param_pre), (name_new, param_new) in zip(old_generator.named_parameters(), new_generator.named_parameters()):
#    if name_pre == name_new and param_pre.shape == param_new.shape:
#        param_new.data.copy_(param_pre.data)

#for (name_pre, param_pre), (name_new, param_new) in zip(old_discriminator.named_parameters(), new_discriminator.named_parameters()):
#    if name_pre == name_new and param_pre.shape == param_new.shape:
#        param_new.data.copy_(param_pre.data)

# Uncomment the lines below when you want to load a model to resume training or for inference

#LOAD_EPOCH = 445  # replace XX with the epoch number you want to load
#new_generator.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"generator_epoch_{LOAD_EPOCH}.pth")))
#new_discriminator.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"discriminator_epoch_{LOAD_EPOCH}.pth")))

#state_dict = old_generator.state_dict()
#new_generator.load_state_dict(state_dict, strict=False)

#state_dict_discriminator = old_discriminator.state_dict()
#new_discriminator.load_state_dict(state_dict_discriminator, strict=False)

# Define Loss and Optimizers
criterion = nn.BCELoss()
optimizer_g = torch.optim.Adam(new_generator.parameters(), lr=lr, betas=betas)
optimizer_d = torch.optim.Adam(new_discriminator.parameters(), lr=lr, betas=betas)

SAVE_INTERVAL = 5  # You can adjust this based on how often you want to save

os.makedirs(MODEL_DIR, exist_ok=True)

#print("Starting epoch loop")
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    
    for i, (images, labels) in enumerate(dataloader):

        #print("Inside dataloader loop")

        images = images.to('xpu')

        current_batch_size = images.size(0)

        real_label_value = 0.9
        fake_label_value = 0.1

        real_labels = torch.full((current_batch_size, 1), real_label_value).to('xpu')
        fake_labels = torch.full((current_batch_size, 1), fake_label_value).to('xpu')

        # Training Discriminator
        optimizer_d.zero_grad()

        outputs = new_discriminator(images).view(-1, 1)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Generate fake images
        z = torch.randn(current_batch_size, z_dim, 1, 1).to('xpu')
        fake_images = new_generator(z)

        outputs = new_discriminator(fake_images.detach()).view(-1, 1)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Training Generator
        optimizer_g.zero_grad()
        
        z = torch.randn(current_batch_size, z_dim, 1, 1).to('xpu')
        fake_images = new_generator(z)
        outputs = new_discriminator(fake_images).view(-1, 1)
        loss_g = criterion(outputs, real_labels)
        loss_g.backward()
        optimizer_g.step()

        # for _ in range(2):  # Update the Generator twice
        #     # Training Generator
        #     optimizer_g.zero_grad()

        #     z = torch.randn(current_batch_size, z_dim).to('xpu')
        #     fake_images = generator(z)
        #     outputs = discriminator(fake_images)
        #     loss_g = criterion(outputs, real_labels)

        #     if _ == 0:
        #         loss_g.backward(retain_graph=True)
        #     else:
        #         loss_g.backward()

        #     #print(f"loss_g: {loss_g.item()}")
        #     #print(f"loss_d: {loss_d.item()}")
        #     #loss_g.backward()  # Backpropagation for the Generator
        #     optimizer_g.step()  # Update the Generator's weights

        # print every 100 batches
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss_D: {d_loss.item()}, Loss_G: {loss_g.item()}")

        if i % 500 == 0:
            save_sample(epoch, fixed_input, new_generator)
    
    if epoch % SAVE_INTERVAL == 0:
        generator_filename = os.path.join(MODEL_DIR, f"generator_epoch_{epoch+445}.pth")
        discriminator_filename = os.path.join(MODEL_DIR, f"discriminator_epoch_{epoch+445}.pth")
        
        # Save both generator and discriminator checkpoints
        torch.save(new_generator.state_dict(), generator_filename)
        torch.save(new_discriminator.state_dict(), discriminator_filename)
        print(f"Saved models for epoch {epoch}")


