import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd

# Create output directories
os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per generator iteration")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
parser.add_argument("--dataset", type=str, default="mnist", help="dataset to use (mnist, cifar10, fashion-mnist)")
parser.add_argument("--wgan_type", type=str, default="gp", help="WGAN type (vanilla, gp)")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Calculate final output size based on image dimensions
        self.final_size = int(np.prod(img_shape))  # Will be 784 for MNIST or 3072 for CIFAR-10

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, self.final_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        # Calculate input size based on image dimensions
        self.input_size = int(np.prod(img_shape))  # Will be 784 for MNIST or 3072 for CIFAR-10

        self.model = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

class WGAN:
    def __init__(self):
        self.generator = Generator().to(device)
        self.critic = Critic().to(device)

        # Configure data loader
        if opt.dataset == "mnist":
            self.dataloader = torch.utils.data.DataLoader(
                datasets.MNIST(
                    "../../data/mnist",
                    train=True,
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize(opt.img_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]),
                ),
                batch_size=opt.batch_size,
                shuffle=True,
            )
        elif opt.dataset == "cifar10":
            self.dataloader = torch.utils.data.DataLoader(
                datasets.CIFAR10(
                    "../../data/cifar10",
                    train=True,
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize((opt.img_size, opt.img_size)),  # Resize while maintaining aspect ratio
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ]),
                ),
                batch_size=opt.batch_size,
                shuffle=True,
            )
        elif opt.dataset == "fashion-mnist":
            self.dataloader = torch.utils.data.DataLoader(
                datasets.FashionMNIST(
                    "../../data/fashion-mnist",
                    train=True,
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize(opt.img_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]),
                ),
                batch_size=opt.batch_size,
                shuffle=True,
            )

        # Optimizers
        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=opt.lr)
        self.optimizer_D = torch.optim.RMSprop(self.critic.parameters(), lr=opt.lr)

    def train_vanilla(self):
        batches_done = 0
        for epoch in range(opt.n_epochs):
            for i, (imgs, _) in enumerate(self.dataloader):
                # Configure input
                real_imgs = imgs.to(device)

                # ---------------------
                #  Train Critic
                # ---------------------
                for _ in range(opt.n_critic):
                    self.optimizer_D.zero_grad()

                    # Sample noise as generator input
                    z = torch.randn(imgs.shape[0], opt.latent_dim, device=device)

                    # Generate a batch of images
                    fake_imgs = self.generator(z).detach()
                    # Adversarial loss
                    loss_D = -torch.mean(self.critic(real_imgs)) + torch.mean(self.critic(fake_imgs))

                    loss_D.backward()
                    self.optimizer_D.step()

                    # Clip weights of critic
                    for p in self.critic.parameters():
                        p.data.clamp_(-opt.clip_value, opt.clip_value)

                # -----------------
                #  Train Generator
                # -----------------
                self.optimizer_G.zero_grad()
                # Generate fresh noise for generator
                z = torch.randn(imgs.shape[0], opt.latent_dim, device=device)
                gen_imgs = self.generator(z)
                loss_G = -torch.mean(self.critic(gen_imgs))

                loss_G.backward()
                self.optimizer_G.step()

                if batches_done % opt.sample_interval == 0:
                    # Print progress
                    print(
                        f"[Epoch {epoch}/{opt.n_epochs}] [Batch {batches_done % len(self.dataloader)}/{len(self.dataloader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]"
                    )
                    # Save images
                    save_image(gen_imgs.data[:25], 
                              f"images/{opt.dataset}_epoch_{epoch}_batch_{batches_done}.png", 
                              nrow=5, 
                              normalize=True,
                              padding=2 if opt.channels == 3 else 0)
                    torch.save({
                        'generator_state_dict': self.generator.state_dict(),
                        'critic_state_dict': self.critic.state_dict(),
                        'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                        'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                    }, f"saved_models/checkpoint_{batches_done}.pt")
                batches_done += 1

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        validity = self.critic(interpolates)
        fake = torch.ones(validity.shape, device=device)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=validity,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_wgan_gp(self):
        lambda_gp = 10  # Gradient penalty lambda hyperparameter
        
        batches_done = 0
        for epoch in range(opt.n_epochs):
            for i, (imgs, _) in enumerate(self.dataloader):
                # Configure input
                real_imgs = imgs.to(device)

                # ---------------------
                #  Train Critic
                # ---------------------
                for _ in range(opt.n_critic):
                    self.optimizer_D.zero_grad()

                    # Sample noise as generator input
                    z = torch.randn(imgs.shape[0], opt.latent_dim, device=device)

                    # Generate a batch of images
                    fake_imgs = self.generator(z).detach()

                    # Real images
                    real_validity = self.critic(real_imgs)
                    # Fake images
                    fake_validity = self.critic(fake_imgs)
                    # Gradient penalty
                    gradient_penalty = self.compute_gradient_penalty(real_imgs.data, fake_imgs.data)
                    # Adversarial loss
                    loss_D = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

                    loss_D.backward()
                    self.optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------
                self.optimizer_G.zero_grad()
                # Generate fresh noise for generator
                z = torch.randn(imgs.shape[0], opt.latent_dim, device=device)
                gen_imgs = self.generator(z)
                loss_G = -torch.mean(self.critic(gen_imgs))

                loss_G.backward()
                self.optimizer_G.step()

                if batches_done % opt.sample_interval == 0:
                    # Print progress
                    print(
                        f"[Epoch {epoch}/{opt.n_epochs}] [Batch {batches_done % len(self.dataloader)}/{len(self.dataloader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]"
                    )
                    # Save images
                    save_image(gen_imgs.data[:25], 
                              f"images/{opt.dataset}_epoch_{epoch}_batch_{batches_done}.png", 
                              nrow=5, 
                              normalize=True,
                              padding=2 if opt.channels == 3 else 0)
                
                    
                batches_done += 1
            torch.save({
                'generator_state_dict': self.generator.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            }, f"saved_models/checkpoint_epoch_{epoch}_gp.pt")

if __name__ == "__main__":
    wgan = WGAN()
    if opt.wgan_type == "vanilla":
        wgan.train_vanilla()
    else:
        wgan.train_wgan_gp()

