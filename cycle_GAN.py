import os
import numpy as np
import itertools
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from huggingface_hub import HfApi, HfFolder, Repository

# define the Device
device = torch.device("cpu")  # No need to check for CUDA

# Define the Generator and Discriminator architectures
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# Define the dataset class
class AstroDataset(Dataset):
    def __init__(self, raw_dir, hubble_dir, transform=None):
        self.raw_dir = raw_dir
        self.hubble_dir = hubble_dir
        self.transform = transform
        self.raw_files = [
            os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".tif")
        ]
        self.hubble_files = [
            os.path.join(hubble_dir, f)
            for f in os.listdir(hubble_dir)
            if f.endswith(".jpg")
        ]

        print(f"Found {len(self.raw_files)} raw TIF images.")
        print(f"Found {len(self.hubble_files)} hubble jpg images.")
        assert len(self.raw_files) > 0, "No raw images found in the directory."
        assert (
            len(self.hubble_files) > 0
        ), "No hubble images found in the directory."

    def __len__(self):
        return min(len(self.raw_files), len(self.hubble_files))

    def __getitem__(self, idx):
        raw_path = self.raw_files[idx]
        hubble_path = self.hubble_files[idx]

        raw_image = Image.open(raw_path).convert("RGB")
        hubble_image = Image.open(hubble_path).convert("RGB")

        if self.transform:
            raw_image = self.transform(raw_image)
            hubble_image = self.transform(hubble_image)

        return {"raw": raw_image, "hubble": hubble_image}


# Define the training function
def train(
    G_AB,
    G_BA,
    D_A,
    D_B,
    optimizer_G,
    optimizer_D_A,
    optimizer_D_B,
    criterion_GAN,
    criterion_cycle,
    criterion_identity,
    dataloader,
    num_epochs=50,
):

    G_AB.to(device)
    G_BA.to(device)
    D_A.to(device)
    D_B.to(device)

    G_losses = []
    D_A_losses = []
    D_B_losses = []

    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            real_A = batch["raw"].to(device)
            real_B = batch["hubble"].to(device)

            # Generate fake images
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)

            # Resize to match the output of discriminators
            valid_size = D_B(fake_B).size()[2:]
            valid = torch.ones((real_A.size(0), 1, *valid_size)).to(device)
            fake = torch.zeros((real_A.size(0), 1, *valid_size)).to(device)

            # Discriminator A
            optimizer_D_A.zero_grad()
            loss_D_A_real = criterion_GAN(D_A(real_A), valid)
            loss_D_A_fake = criterion_GAN(D_A(fake_A.detach()), fake)
            loss_D_A = (loss_D_A_real + loss_D_A_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            # Discriminator B
            optimizer_D_B.zero_grad()
            loss_D_B_real = criterion_GAN(D_B(real_B), valid)
            loss_D_B_fake = criterion_GAN(D_B(fake_B.detach()), fake)
            loss_D_B = (loss_D_B_real + loss_D_B_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            # Generators
            optimizer_G.zero_grad()
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            # Cycle loss
            recov_A = G_BA(fake_B)
            recov_B = G_AB(fake_A)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            # Identity loss
            identity_A = G_BA(real_A)
            identity_B = G_AB(real_B)
            loss_identity_A = criterion_identity(identity_A, real_A)
            loss_identity_B = criterion_identity(identity_B, real_B)

            # Total loss
            loss_G = (
                loss_GAN_AB
                + loss_GAN_BA
                + 10.0 * (loss_cycle_A + loss_cycle_B)
                + 5.0 * (loss_identity_A + loss_identity_B)
            )
            loss_G.backward()
            optimizer_G.step()

            # Record losses
            G_losses.append(loss_G.item())
            D_A_losses.append(loss_D_A.item())
            D_B_losses.append(loss_D_B.item())

            # Print loss values
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}] "
                f"Loss D_A: {loss_D_A.item():.4f}, Loss D_B: {loss_D_B.item():.4f}, "
                f"Loss G: {loss_G.item():.4f}"
            )

    return G_losses, D_A_losses, D_B_losses


# Define transformations
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

# Setup dataset and dataloader
raw_dir = "C:/Users.........RAW_hubble_image"
hubble_dir = "C:/Users.....hubble_processed"
dataset = AstroDataset(raw_dir=raw_dir, hubble_dir=hubble_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True) #you can set how much you want batch_size


# Initialize models, optimizers, and loss functions
G_AB = Generator()
G_BA = Generator()
D_A = Discriminator()
D_B = Discriminator()

optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Start training
G_losses, D_A_losses, D_B_losses = train(
    G_AB,
    G_BA,
    D_A,
    D_B,
    optimizer_G,
    optimizer_D_A,
    optimizer_D_B,
    criterion_GAN,
    criterion_cycle,
    criterion_identity,
    dataloader,
    num_epochs=50,
)

# Plot the losses
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_A_losses, label="D_A")
plt.plot(D_B_losses, label="D_B")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# (Optional) Save the models
torch.save(G_AB.state_dict(), "G_AB.pth")
torch.save(G_BA.state_dict(), "G_BA.pth")
torch.save(D_A.state_dict(), "D_A.pth")
torch.save(D_B.state_dict(), "D_B.pth")

# (Optional) Test with sample images
sample_data = next(iter(dataloader))
raw_image = sample_data["raw"].to(device)
hubble_image = sample_data["hubble"].to(device)

# Generate translated images
fake_hubble = G_AB(raw_image)
fake_raw = G_BA(hubble_image)

# Convert to numpy for visualization
raw_image_np = raw_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
hubble_image_np = hubble_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
fake_hubble_np = fake_hubble.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
fake_raw_np = fake_raw.squeeze().cpu().detach().numpy().transpose(1, 2, 0)

# Plot the images
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax[0, 0].imshow(raw_image_np)
ax[0, 0].set_title("RAW Image")
ax[0, 1].imshow(fake_hubble_np)
ax[0, 1].set_title("Fake Hubble Image")
ax[1, 0].imshow(hubble_image_np)
ax[1, 0].set_title("Hubble Image")
ax[1, 1].imshow(fake_raw_np)
ax[1, 1].set_title("Fake RAW Image")
plt.show()

''' #if you want to upload on HF (optional )
from huggingface_hub import HfApi, HfFolder, Repository


# Replace 'your_access_token' with your actual token
access_token = "hf_xxxxxxxxxxxxxxxxxxx"
# Use HfApi to interact with the Hugging Face Hub

# Use the Repository class to clone and push to your repository
repo = Repository(
    local_dir="cycleGAN_model",
    clone_from="hf_username/hf_reponame",
    use_auth_token=access_token,
)

# Save models to the repository folder
torch.save(G_AB.state_dict(), os.path.join("cycleGAN_model", "G_AB.pth"))
torch.save(G_BA.state_dict(), os.path.join("cycleGAN_model", "G_BA.pth"))
torch.save(D_A.state_dict(), os.path.join("cycleGAN_model", "D_A.pth"))
torch.save(D_B.state_dict(), os.path.join("cycleGAN_model", "D_B.pth"))

# Push to Hugging Face
repo.push_to_hub()

print("Model uploaded to Hugging Face Hub.")
 '''
