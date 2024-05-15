import jittor as jt
import jittor.transform as transform
from jittor import nn
from jittor.dataset.mnist import MNIST
import numpy as np
import cv2
import os
import optparse


jt.flags.use_cuda = False

# Define hyperparameters
input_size = 784
hidden_size1 = 512
hidden_size2 = 256
latent_size = 64
n_epochs = 10
sample_interval = 400
beta = 0.1

# parse beta from command line
parser = optparse.OptionParser()
parser.add_option('-b', '--beta', dest='beta', default=0.1, type='float', help='beta for beta-vae')
options, args = parser.parse_args()
beta = options.beta

# Create directories
os.makedirs(f"../intermediate/beta-vae-{beta}", exist_ok=True)

# Save images as nrow x nrow grids
def save_image(img, path, nrow):
    img = img.reshape(nrow * nrow, 1, 28, 28) # (25, 1, 28, 28)
    # img.view(-1, 1, 28, 28) # (25, 1, 28, 28)
    N,C,H,W = img.shape
    img2 = np.zeros((H*nrow, W*nrow, C), dtype=img.dtype)
    for i in range(nrow):
        for j in range(nrow):
            img2[i*H:(i+1)*H, j*W:(j+1)*W,:] = img[i*nrow+j].transpose(1,2,0)
    img2 = img2 * 255.0
    cv2.imwrite(path, img2)

# Define VAE model
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, latent_size):
        super(VAE, self).__init__()

        # Encoder part
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc31 = nn.Linear(hidden_size2, latent_size) # mean
        self.fc32 = nn.Linear(hidden_size2, latent_size) # logvar

        # Decoder part
        self.fc4 = nn.Linear(latent_size, hidden_size2)
        self.fc5 = nn.Linear(hidden_size2, hidden_size1)
        self.fc6 = nn.Linear(hidden_size1, input_size)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.latent_size = latent_size

    def encode(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, logvar
    
    def decode(self, z):
        h = self.relu(self.fc4(z))
        h = self.relu(self.fc5(h))
        return self.sigmoid(self.fc6(h))
    
    def sample(self, mu, logvar):
        std = jt.exp(0.5*logvar)
        # eps = jt.randn([mu.shape[0], self.latent_size])
        eps = jt.randn(std.shape)
        z = mu + eps * std
        return z

    def execute(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.sample(mu, logvar)
        return self.decode(z), mu, logvar

# Instantiate dataloader
train_loader = MNIST(train=True, transform=transform.Gray()).set_attrs(batch_size=25, shuffle=True)
val_loader = MNIST(train=False, transform=transform.Gray()).set_attrs(batch_size=25, shuffle=False)

# Instantiate VAE and configure optimizer
vae = VAE(input_size, hidden_size1, hidden_size2, latent_size)
optimizer = nn.Adam(vae.parameters(), lr=1e-5)

if jt.has_cuda:
    vae = vae.cuda()
    optimizer = optimizer.cuda()

# Define loss function using MSE loss and KL divergence
def loss_function(recon_x, x, mu, logvar , beta=0.1):
    # BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    MSE = jt.mean((recon_x - x.view(-1, 784)) ** 2)
    KLD = -0.5 * jt.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD

# Define train function
def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    lens = len(train_loader)
    for batch_idx, (inputs, _) in enumerate(train_loader):
        if jt.has_cuda:
            inputs = inputs.cuda()
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(inputs)
        loss = loss_function(recon_batch, inputs, mu, logvar, beta=beta)

        optimizer.step(loss)
        train_loss += loss.data[0]
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), lens * train_loader.batch_size,
                100. * batch_idx / lens, loss.data[0] / len(inputs)))
        batches_done = epoch * lens + batch_idx
        if batches_done % sample_interval == 0:
            z = jt.randn([25, latent_size])
            if jt.has_cuda:
                z = z.cuda()
            sample = vae.decode(z)
            # Save sample images and reconstruction images
            save_image(sample.data[:25], f'../intermediate/beta-vae-{beta}/{batches_done}_sample.png', nrow=5)
            save_image(recon_batch.data[:25], f'../intermediate/beta-vae-{beta}/{batches_done}_recon.png', nrow=5)
        
    print('====> Epoch: {}/{} Average loss: {:.4f}'.format(epoch, n_epochs, train_loss/lens))

# Define val function
def val(model, val_loader, epoch):
    model.eval()
    val_loss = 0
    with jt.no_grad():
        for inputs, _ in val_loader:
            if jt.has_cuda:
                inputs = inputs.cuda()
            recon_batch, mu, logvar = model(inputs)
            val_loss += loss_function(recon_batch, inputs, mu, logvar, beta=beta).data[0]

    val_loss /= len(val_loader.dataset)
    print('====> Epoch: {} Val set loss: {:.4f}'.format(epoch, val_loss))

# Train and val
for epoch in range(n_epochs):
    train(vae, train_loader, optimizer, epoch)
    val(vae, val_loader, epoch)

# Save model
vae.save(f'../ckpts/beta-vae/beta-vae-{beta}.model')

# Final sample
with jt.no_grad():
    z = jt.randn([100, latent_size])
    if jt.has_cuda:
        z = z.cuda()
    sample = vae.decode(z)
    save_image(sample.data[:100], f'../results/beta-vae-{beta}.png', nrow=10)