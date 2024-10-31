import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from torch.autograd import Variable, grad

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

# Generator Network
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        return self.main(input)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)

# Compute gradient penalty
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size()).to(device)
    # Get gradient w.r.t. interpolates
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', 
                        required=True, 
                        help='cifar10 | mnist | folder', 
                        default='cifar10')
    parser.add_argument('--dataroot', 
                        required=True, 
                        help='path to dataset', 
                        default="images")
    parser.add_argument('--workers', 
                        type=int, 
                        default=8, 
                        help='number of data loading workers')
    parser.add_argument('--batchSize', 
                        type=int, 
                        default=64, 
                        help='input batch size')
    parser.add_argument('--imageSize', 
                        type=int, 
                        default=32, 
                        help='the height / width of the input image to network')
    parser.add_argument('--nz', 
                        type=int, 
                        default=100, 
                        help='size of the latent z vector')
    parser.add_argument('--ngf', 
                        type=int, 
                        default=64, 
                        help='number of generator filters')
    parser.add_argument('--ndf', 
                        type=int, 
                        default=64, 
                        help='number of discriminator filters')
    parser.add_argument('--niter', 
                        type=int, 
                        default=25, 
                        help='number of epochs to train for')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.0002, 
                        help='learning rate')
    parser.add_argument('--beta1', 
                        type=float, 
                        default=0.5, 
                        help='beta1 for adam')
    parser.add_argument('--gpu', 
                        action='store_true', 
                        help='enables GPU acceleration (CUDA or MPS)')
    parser.add_argument('--outf', 
                        default='output', 
                        help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', 
                        type=int, 
                        help='manual seed')
    parser.add_argument('--lambda_gp', 
                        type=float, 
                        default=10, 
                        help='gradient penalty lambda hyperparameter')
    parser.add_argument('--lr_d', 
                        type=float, 
                        default=0.0001, 
                        help='learning rate for discriminator')
    parser.add_argument('--lr_g', 
                        type=float, 
                        default=0.0001, 
                        help='learning rate for generator')
    parser.add_argument('--n_critic', 
                        type=int, 
                        default=5, 
                        help='number of discriminator iterations per generator iteration')
    
    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed:", opt.manualSeed)
    set_seed(opt.manualSeed)

    # Replace the device selection logic
    if torch.backends.mps.is_available() and opt.gpu:
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available() and opt.gpu:
        device = torch.device("cuda:0")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Dataset
    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        dataset = dset.ImageFolder(root=opt.dataroot,
                                 transform=transforms.Compose([
                                     transforms.Resize(opt.imageSize),
                                     transforms.CenterCrop(opt.imageSize),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 ]))
        nc = 3
    elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc = 1
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                              transform=transforms.Compose([
                                  transforms.Resize(opt.imageSize),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ]))
        nc = 3

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                           shuffle=True, num_workers=opt.workers)

    # Create the generator and discriminator
    netG = Generator(opt.nz, opt.ngf, nc).to(device)
    netD = Discriminator(opt.ndf, nc).to(device)

    # Initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

    # Training Loop
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize D(x) - D(G(z))
            ###########################
            # Train critic for n_critic iterations
            for _ in range(opt.n_critic):
                netD.zero_grad()
                
                real_data = data[0].to(device)
                batch_size = real_data.size(0)
                
                # Train with real and fake data
                d_real = netD(real_data).view(-1)
                d_real = d_real.mean()

                noise = torch.randn(batch_size, opt.nz, 1, 1, device=device)
                fake = netG(noise)
                d_fake = netD(fake.detach()).view(-1)
                d_fake = d_fake.mean()

                gradient_penalty = compute_gradient_penalty(netD, real_data, fake.detach(), device)
                
                d_loss = -d_real + d_fake + opt.lambda_gp * gradient_penalty
                
                d_loss.backward()
                optimizerD.step()

            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################
            netG.zero_grad()
            
            # Generate new fake data for generator update
            noise = torch.randn(batch_size, opt.nz, 1, 1, device=device)
            fake = netG(noise)
            g_loss = -netD(fake).mean()
            
            g_loss.backward()
            optimizerG.step()

            # Output training stats
            if i % 100 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (epoch, opt.niter, i, len(dataloader),
                         d_loss.item(), g_loss.item()))

            # Save generated images
            if i % 100 == 0:
                vutils.save_image(real_data,
                                '%s/real_samples.png' % opt.outf,
                                normalize=True)
                fake = netG(torch.randn(batch_size, opt.nz, 1, 1, device=device))
                vutils.save_image(fake.detach(),
                                f'{opt.outf}/fake_samples_epoch_{epoch}_batch_{i}.png',
                                normalize=True)

        # Save models
        if epoch % 5 == 0:
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

if __name__ == '__main__':
    main()
