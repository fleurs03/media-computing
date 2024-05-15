import jittor as jt
from jittor import nn
from jittor.dataset.mnist import MNIST
import jittor.transform as transform
import numpy as np
import cv2
import os

jt.flags.use_cuda = False

# Define hyperparameters
n_epochs = 200
batch_size = 64
lr = 5e-5
n_cpu = 8
latent_size = 100
img_size = 28
channels = 1
n_critic = 5
clip_value = 0.01
sample_interval = 400

# Create directories
os.makedirs("../intermediate/wgan", exist_ok=True)
img_shape = (channels, img_size, img_size)

# Save images as nrow x nrow grids
def save_image(img, path, nrow):
    N,C,H,W = img.shape
    img2 = np.zeros((H*nrow, W*nrow, C), dtype=img.dtype)
    for i in range(nrow):
        for j in range(nrow):
            img2[i*H:(i+1)*H, j*W:(j+1)*W,:] = img[i*nrow+j].transpose(1,2,0)
    img2 = (img2 + 1.0) / 2.0 * 255.0
    cv2.imwrite(path, img2)

# Clamp to avoid gradient exploding
def clamp(var, l, r):
    var.assign(jt.minimum(jt.maximum(var, l), r))

# Define Generator
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # Wrap model block
        def block(in_size, out_size, normalize=True):
            layers = [nn.Linear(in_size, out_size)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_size, 0.1))
            layers.append(nn.LeakyReLU(scale=0.01))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
            )

    def execute(self, z):
        img = self.model(z)
        return img.view((img.shape[0], *img_shape))
    
# Define Discriminator
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(scale=0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(scale=0.01),
            nn.Linear(256, 1)
            )

    def execute(self, img):
        img = img.view((img.shape[0], (-1)))
        return self.model(img)
    
# Instantiate generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Configure data loader
transform = transform.Compose([
    transform.Resize(img_size),
    transform.Gray(),
    transform.ImageNormalize(mean=[0.5], std=[0.5]),
])
dataloader = MNIST(train=True, transform=transform).set_attrs(batch_size=batch_size, shuffle=True, num_workers=n_cpu)

#Configure RMSprop optimizers
optimizer_G = nn.RMSprop(generator.parameters(), lr=lr)
optimizer_D = nn.RMSprop(discriminator.parameters(), lr=lr)

for epoch in range(n_epochs):
    for i, (inputs, _) in enumerate(dataloader):

        # train discriminator
        z = jt.array(np.random.normal(0, 1, (inputs.shape[0], latent_size))).float32()
        gen_imgs = generator(z).detach()
        real_validity = discriminator(inputs)
        fake_validity = discriminator(gen_imgs)
        d_loss = (-(jt.mean(real_validity) - jt.mean(fake_validity)))
        optimizer_D.step(d_loss)
        for p in discriminator.parameters():
            clamp(p, (- clip_value), clip_value)
        
        # train generator
        if (i % n_critic) == 0:
            gen_imgs = generator(z)
            fake_validity = discriminator(gen_imgs)
            g_loss = (-(jt.mean(fake_validity)))
            optimizer_G.step(g_loss)
            print(
                ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, n_epochs, i, len(dataloader), d_loss.data[0], g_loss.data[0]))
            )

        batches_done = ((epoch * len(dataloader)) + i)
        if (batches_done % sample_interval) == 0:
            # Save generated images
            save_image(gen_imgs.data[:25], (f"../intermediate/wgan/{batches_done}.png"), nrow=5)

# Save models
generator.save('../ckpts/wgan/generator.model')
discriminator.save('../ckpts/wgan/discriminator.model')

# Final sample
with jt.no_grad():
    z = jt.randn([100, latent_size])
    if jt.has_cuda:
        z = z.cuda()
    sample = generator(z)
    save_image(sample.data[:100], f'../results/wgan.png', nrow=10)


