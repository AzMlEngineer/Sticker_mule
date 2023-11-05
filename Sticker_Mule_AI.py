#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import required libraries
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
CODING_SIZE = 100
BATCH_SIZE = 32
IMAGE_SIZE = 64
device = torch.device("cuda:0" if
torch.cuda.is_available() else "cpu")

#loads the training data, defines the transforms, and creates a dataloader for batch iteration
transform = transforms.Compose([
 transforms.Resize(IMAGE_SIZE),
 transforms.ToTensor(),
])
dataset = datasets.FashionMNIST(
 './',
 train=True,
 download=True,
 transform=transform)
dataloader = DataLoader(
 dataset,
 batch_size=BATCH_SIZE,
 shuffle=True,
 num_workers=8)


# In[ ]:


#display a batch of images
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
data_batch, labels_batch = next(iter(dataloader))
grid_img = make_grid(data_batch, nrow=8)
plt.imshow(grid_img.permute(1, 2, 0))


# In[ ]:


#Designed to create an image from an input vector of 100 random values
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self, coding_sz):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(coding_sz,
                               1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024,
                               512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128,1, 4, 2, 1),
            nn.Tanh()
       )
    def forward(self, input):
        return self.net(input)
netG = Generator(CODING_SIZE).to(device)


# In[ ]:


#create the Discriminator module for determines the probability that the input image is real
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,
              self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, 4, 1, 0),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        return self.net(input)
netD = Discriminator().to(device)


# In[ ]:


#initialize the weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
netG.apply(weights_init)
netD.apply(weights_init)


# In[ ]:


#Define the loss function and optimizers that will be used to train the generator and the discriminator
from torch import optim
criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(),
                        lr=0.0002,
                        betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(),
                        lr=0.0001,
                        betas=(0.5, 0.999))
#Define values for the real and fake labels and create tensors for computing the loss
real_labels = torch.full((BATCH_SIZE,),
                         1.,
                         dtype=torch.float,
                         device=device)
fake_labels = torch.full((BATCH_SIZE,),
                         0.,
                         dtype=torch.float,
                         device=device)


# In[ ]:


# create lists for storing the errors and define a test vector
G_losses = []
D_losses = []
D_real = []
D_fake = []
z = torch.randn((
    BATCH_SIZE, 100)).view(-1, 100, 1, 1).to(device)
test_out_images = []


# In[ ]:


#Training loop
N_EPOCHS = 5
for epoch in range(N_EPOCHS):
    print(f'Epoch: {epoch}')
    for i, batch in enumerate(dataloader):
        if (i%200==0):
            print(f'batch: {i} of {len(dataloader)}')
        # Train Discriminator with an all-real batch.
        netD.zero_grad()
        real_images = batch[0].to(device) *2. - 1.
        #pass real images to the Discriminator
        output = netD(real_images).view(-1)
        errD_real = criterion(output, real_labels)
        D_x = output.mean().item()
        # Train Discriminator with an all-fake batch.
        noise = torch.randn((BATCH_SIZE,
                             CODING_SIZE))
        noise = noise.view(-1,100,1,1).to(device)
        fake_images = netG(noise)
        #pass fake images to the Discriminator
        output = netD(fake_images).view(-1)
        errD_fake = criterion(output, fake_labels)
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        #run backpropagation and update the Discriminator
        errD.backward(retain_graph=True)
        optimizerD.step()
        # Train Generator to generate better fakes.
        netG.zero_grad()
        #pass fake images to the updated Discriminator
        output = netD(fake_images).view(-1)
        #the Generator loss is based on cases in which the Discriminator is wrong.
        errG = criterion(output, real_labels)
        #run backpropagation and update the Generator
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        # Save losses for plotting later.
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        D_real.append(D_x)
        D_fake.append(D_G_z2)
    #create a batch of images and save them after each epoch
    test_images = netG(z).to('cpu').detach()
    test_out_images.append(test_images


# In[ ]:


#save trained model for deployment
torch.save(netG.state_dict(), './gan.pt')


# In[ ]:




