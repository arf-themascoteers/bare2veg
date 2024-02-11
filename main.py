from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from generator import Generator
from discriminator import Discriminator
import os
import utils
from torch.utils.data import DataLoader
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_train = pd.read_csv("data/train.csv").sample(frac=1).to_numpy()
dataset_train = torch.tensor(dataset_train, dtype=torch.float32).to(device)

dataset_test = pd.read_csv("data/test.csv").to_numpy()
dataset_test = torch.tensor(dataset_test, dtype=torch.float32).to(device)

loss_comparison = nn.BCEWithLogitsLoss()
L1_loss = nn.L1Loss()

def discriminator_training(inputs, targets, discriminator_opt):
    discriminator_opt.zero_grad()
    output = discriminator(inputs, targets)
    label = torch.ones(size=output.shape, dtype=torch.float, device=device)
    real_loss = loss_comparison(output, label)
    gen_image = generator(inputs).detach()
    fake_output = discriminator(inputs, gen_image)
    fake_label = torch.zeros(size=fake_output.shape, dtype=torch.float, device=device)
    fake_loss = loss_comparison(fake_output, fake_label)
    Total_loss = (real_loss + fake_loss) / 2
    Total_loss.backward()
    discriminator_opt.step()
    return Total_loss


def generator_training(inputs, targets, generator_opt, L1_lambda):
    generator_opt.zero_grad()
    generated_image = generator(inputs)
    disc_output = discriminator(inputs, generated_image)
    desired_output = torch.ones(size=disc_output.shape, dtype=torch.float, device=device)
    generator_loss = loss_comparison(disc_output, desired_output) + L1_lambda * torch.abs(
        generated_image - targets).sum()
    generator_loss.backward()
    generator_opt.step()

    return generator_loss, generated_image

L1_lambda = 100
NUM_EPOCHS= 3
lr=0.0002
beta1=0.5
beta2=0.999

discriminator = Discriminator()
generator = Generator()

discriminator_opt = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
generator_opt = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

discriminator = discriminator.to(device)
generator = generator.to(device)

for epoch in range(NUM_EPOCHS):
    print(f"Training epoch {epoch+1}")
    inputs = dataset_train[:,1:8]
    targets = dataset_train[:,9:]

    Disc_Loss = discriminator_training(inputs,targets,discriminator_opt)
    for i in range(2):
        Gen_Loss, generator_image = generator_training(inputs,targets, generator_opt, L1_lambda)

    if (epoch % 10) == 0:
         utils.print_images(inputs,5)
         utils.print_images(generator_image,5)
         utils.print_images(targets,5)

inputs = dataset_test[:, 1:8]
targets = dataset_test[:, 9:]
gen = generator(inputs)
satellite = inputs.detach().cpu()
gen = gen.detach().cpu()
maps = targets.detach().cpu()

utils.print_images(satellite,10)
utils.print_images(gen,10)
utils.print_images(maps,10)