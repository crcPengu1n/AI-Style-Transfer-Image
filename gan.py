from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

NOISE_DIM = 96


def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device="cpu"):
    """
    Generate a PyTorch Tensor of random noise from Gaussian distribution.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, noise_dim) containing
      noise from a Gaussian distribution.
    """
    noise = None

    noise = torch.randn(batch_size, noise_dim, dtype=dtype, device=device)


    return noise


def discriminator():
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None

    # Replace "pass" statement with your code
    model = nn.Sequential(
        # Fully connected layer from 784 to 400
        nn.Linear(784, 400),
        # LeakyReLU with alpha = 0.05
        nn.LeakyReLU(0.05),

        # Fully connected layer from 400 to 200
        nn.Linear(400, 200),
        # LeakyReLU with alpha = 0.05
        nn.LeakyReLU(0.05),

        # Fully connected layer from 200 to 100
        nn.Linear(200, 100),
        # LeakyReLU with alpha = 0.05
        nn.LeakyReLU(0.05),

        # Fully connected layer from 100 to 1
        nn.Linear(100, 1)
    )
    return model


    return model


def generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None

    # Replace "pass" statement with your code
    model = nn.Sequential(
        # Fully connected layer from noise_dim to 128
        nn.Linear(noise_dim, 128),
        # ReLU activation
        nn.ReLU(True),

        # Fully connected layer from 128 to 256
        nn.Linear(128, 256),
        # ReLU activation
        nn.ReLU(True),

        # Fully connected layer from 256 to 512
        nn.Linear(256, 512),
        # ReLU activation
        nn.ReLU(True),

        # Fully connected layer from 512 to 784
        nn.Linear(512, 784),
        # TanH activation to clip the output to [-1, 1]
        nn.Tanh()
    )
    return model


    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None

    # Replace "pass" statement with your code
    # Calculate probabilities on logits
    real_probs = torch.sigmoid(logits_real)
    fake_probs = torch.sigmoid(logits_fake)

    # True labels for real data (1s)
    true_labels = torch.ones_like(logits_real)
    
    # False labels for fake data (0s)
    fake_labels = torch.zeros_like(logits_fake)
    
    # Loss for real data - log(D(x))
    real_loss = -torch.log(real_probs + 1e-8)  # Adding epsilon for numerical stability

    # Loss for fake data - log(1 - D(G(z)))
    fake_loss = -torch.log(1 - fake_probs + 1e-8)  # Adding epsilon for numerical stability

    # Total discriminator loss
    loss = (real_loss + fake_loss).mean()  # Averaging over all examples in the batch

    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None

    # Replace "pass" statement with your code
    # Calculate probabilities on logits
    fake_probs = torch.sigmoid(logits_fake)

    # True labels for fake data (generator tries to fool the discriminator)
    true_labels = torch.ones_like(logits_fake)
    
    # Loss for fake data - log(D(G(z)))
    loss = -torch.log(fake_probs + 1e-8)  # Using log probability for the generator's success
    
    # Returning the mean loss across the batch
    loss = loss.mean()

    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None

    # Replace "pass" statement with your code
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))

    return optimizer


def run_a_gan(D, G, D_solver, G_solver, loader_train, discriminator_loss, generator_loss, device, show_images, plt, show_every=250, 
              batch_size=128, noise_size=96, num_epochs=10):
  """
  Train a GAN!
  
  Inputs:
  - D, G: PyTorch models for the discriminator and generator
  - D_solver, G_solver: torch.optim Optimizers to use for training the
    discriminator and generator.
  - loader_train: the dataset used to train GAN
  - discriminator_loss, generator_loss: Functions to use for computing the generator and
    discriminator loss, respectively.
  - show_every: Show samples after every show_every iterations.
  - batch_size: Batch size to use for training.
  - noise_size: Dimension of the noise to use as input to the generator.
  - num_epochs: Number of epochs over the training dataset to use for training.
  """
  D = D.to(device)
  G = G.to(device)

  iter_count = 0
  for epoch in range(num_epochs):
      for x, _ in loader_train:
          if x.size(0) != batch_size:
              continue

          # Ensure data is on the correct device and is properly reshaped
          x = x.to(device).view(batch_size, -1)  # Reshape input to (batch_size, 784) if needed

          # Process real data to be in the range [-1, 1]
          x = 2 * (x - 0.5)


          D_solver.zero_grad()

          # Generate fake data
          noise = torch.randn(batch_size, noise_size, device=device)
          fake_images = G(noise).detach()

          logits_real = D(x)
          logits_fake = D(fake_images)

          d_total_error = discriminator_loss(logits_real, logits_fake)
          d_total_error.backward()
          D_solver.step()


          G_solver.zero_grad()

          noise = torch.randn(batch_size, noise_size, device=device)
          fake_images = G(noise)

          gen_logits_fake = D(fake_images)

          g_error = generator_loss(gen_logits_fake)
          g_error.backward()
          G_solver.step()

          if (iter_count % show_every == 0):
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.item(),g_error.item()))
            imgs_numpy = fake_images.data.cpu()#.numpy()
            show_images(imgs_numpy[0:16])
            plt.show()
            print()
          iter_count += 1
          if epoch == num_epochs - 1:
            return imgs_numpy    




def build_dc_classifier():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator
    implementing the architecture in the notebook.
    """
    model = None

    # Replace "pass" statement with your code
    model = nn.Sequential(
        nn.Unflatten(1, (1, 28, 28)),
        nn.Conv2d(1, 32, 5, stride=1),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 5, stride=1),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(1024, 4*4*64),
        nn.LeakyReLU(0.01),
        nn.Linear(4*4*64, 1)
    )


    return model


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the DCGAN
    generator using the architecture described in the notebook.
    """
    model = None

    # Replace "pass" statement with your code
    model = nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(True),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 7*7*128),
        nn.ReLU(True),
        nn.BatchNorm1d(7*7*128),
        nn.Unflatten(1, (128, 7, 7)),
        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
        nn.ReLU(True),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
        nn.Tanh(),
        nn.Flatten()
    )


    return model
