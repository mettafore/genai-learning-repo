# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import logging
import os


# %%
def setup_logger(logging_level):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging_level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# %%
import argparse

parser = argparse.ArgumentParser(description="VAE Training Script")
parser.add_argument("--dataset_name", type=str, default="cifar10", help="Name of the dataset")
parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension size")
parser.add_argument("--latent_dim", type=int, default=100, help="Latent dimension size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")  
parser.add_argument("--model_path", type=str, default="../../models/vae.pth", help="Path to the model")
parser.add_argument("--images_path", type=str, default="../../images/vae/", help="Path to the images")
# log level
parser.add_argument("--log_level", type=str, default="INFO", help="Log level")
args = parser.parse_args()




dataset_name = args.dataset_name
hidden_dim = args.hidden_dim
latent_dim = args.latent_dim
lr = args.lr
epochs = args.epochs  
log_level = args.log_level
LOGGER = setup_logger(log_level)
LOGGER.info(args)


# %%
class VAE(nn.Module):
    def __init__(self, input_channels, img_size, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # encoder
        self.fc1 = nn.Linear(input_channels * img_size * img_size   , hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_channels * img_size * img_size)
        
        
    def encode(self, x):
        e = F.relu(self.fc1(x))
        e = F.relu(self.fc2(e))
        return self.fc_mean(e), self.fc_logvar(e)
    
    def decode(self, z):
        d = F.relu(self.fc3(z))
        d = torch.sigmoid(self.fc4(d))
        return d
    
    def reparametrize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparametrize(mean, logvar)
        return self.decode(z), mean, logvar
    
    def loss_function(self, x, x_recon, mean, logvar):
        BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return KLD + BCE, KLD, BCE



# %%

def setup_vae(dataset_name, hidden_dim, latent_dim, lr):
    if dataset_name == 'mnist':
        input_channels = 1
        img_size = 28
    elif dataset_name == 'fashion_mnist':
        input_channels = 1
        img_size = 28
    elif dataset_name == 'cifar10':
        input_channels = 3
        img_size = 32
    else:
        raise ValueError("Invalid dataset name")
    
    model = VAE(input_channels, img_size, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer, input_channels, img_size


def get_device():
    # mps
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device





def data_loader(dataset_name, img_size):        
    if dataset_name == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../../data', train=True, download=True, 
                       transform=transforms.Compose([transforms.Resize(img_size),
                                                     transforms.ToTensor()])),
            batch_size=64, shuffle=True
            )
    elif dataset_name == 'fashion_mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../../data', train=True, download=True, 
                       transform=transforms.Compose([transforms.Resize(img_size),
                                                     transforms.ToTensor()])),
            batch_size=64, shuffle=True
            )
    elif dataset_name == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../../data', train=True, download=True, 
                       transform=transforms.Compose([transforms.Resize(img_size),
                                                     transforms.ToTensor()])),
            batch_size=64, shuffle=True
            )
    else:
        raise ValueError("Invalid dataset name")
    return train_loader

# %%
LOGGER.debug(3*32*32)
LOGGER.debug(64*32*32*3)


# %%

def train(model, input_channels, img_size, optimizer, epochs, train_loader, device):
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            LOGGER.debug(data.shape)
            LOGGER.debug(input_channels * img_size * img_size)
            data = data.view(-1, input_channels * img_size * img_size)
            LOGGER.debug(data.shape)
            optimizer.zero_grad()
            x_recon, mean, logvar = model(data)

            loss, kld, bce = model.loss_function(data, x_recon, mean, logvar)
            loss.backward()
            optimizer.step()


        average_loss = loss.item() / len(train_loader.dataset)
        average_bce = bce.item() / len(train_loader.dataset)
        average_kld = kld.item() / len(train_loader.dataset)

        LOGGER.info(f'Epoch [{epoch+1}/{epochs}], Average Loss: {average_loss}, Average KLD: {average_kld}, Average BCE: {average_bce}')

def save_model(model, optimizer, epoch, model_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)  # Add model_path parameter

def load_model(model_path, model, optimizer):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch

def save_images(model, device, img_size, latent_dim, input_channels, num_images=10, images_path=args.images_path):
    model.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim).to(device)
        images = model.decode(noise)
        images = images.view(num_images, input_channels, img_size, img_size)

    # Create the directory if it doesn't exist
    os.makedirs(images_path, exist_ok=True)
    
    # plot in grid square   
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i+1)
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        plt.imshow(img)
        plt.axis('off')
    
    # Save the figure
    save_path = os.path.join(images_path, f"epoch_{epochs}.png")
    plt.savefig(save_path)
    plt.close()  # Close the figure to free up memory

    LOGGER.info(f"Images saved to {save_path}")

def generate_images(model, device, img_size, input_channels, num_images=10):
    model.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, 100).to(device)
        images = model.decode(noise)
        images = images.view(num_images, input_channels, img_size, img_size)
    
    # plot in grid square   
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i+1)
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
LOGGER.debug("HELLO")
if __name__ == "__main__":
    LOGGER.debug("HELLO")
    dataset_name = args.dataset_name
    hidden_dim = args.hidden_dim
    latent_dim = args.latent_dim
    lr = args.lr
    epochs = args.epochs    
    device = get_device()
    LOGGER.info(f"{dataset_name}, {hidden_dim}, {latent_dim}, {lr}, {epochs}")

    model, optimizer, input_channels, img_size = setup_vae(dataset_name, hidden_dim, latent_dim, lr)
    LOGGER.info(f"Image size: {img_size}")
    model = model.to(device)  # Move model to the correct device
    train_loader = data_loader(dataset_name, img_size)  # Move this line here
    train(model, input_channels, img_size, optimizer, epochs, train_loader, device)

    # Save the model after training
    save_model(model, optimizer, epochs, args.model_path) 
    save_images(model, device, img_size, latent_dim, input_channels, num_images=10, images_path=args.images_path)
