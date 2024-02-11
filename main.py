import torch
import torch.nn as nn
import torch.optim as optim
from generator import Generator
from discriminator import Discriminator
import utils
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
dataset_full = pd.read_csv("data/full.csv")
dataset_full = pd.DataFrame(scaler.fit_transform(dataset_full), columns=dataset_full.columns)
train, test = model_selection.train_test_split(dataset_full, test_size=0.1, random_state=2)
dataset_full.to_csv("results/full_normalized.csv", index=False)
train.to_csv("results/train.csv", index=False)
test.to_csv("results/test.csv", index=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_train = train.to_numpy()
dataset_train = torch.tensor(dataset_train, dtype=torch.float32).to(device)

dataset_test = test.to_numpy()
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
    #print(f"Loss {real_loss.item()} {fake_loss.item()} {Total_loss.item()}")
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
NUM_EPOCHS= 1000
lr=0.01

discriminator = Discriminator()
generator = Generator()

discriminator_opt = optim.Adam(discriminator.parameters(), lr=lr)
generator_opt = optim.Adam(generator.parameters(), lr=lr)

discriminator = discriminator.to(device)
generator = generator.to(device)

for epoch in range(NUM_EPOCHS):
    inputs = dataset_train[:,1:8]
    targets = dataset_train[:,9:]

    Disc_Loss = discriminator_training(inputs,targets,discriminator_opt)
    for i in range(2):
        Gen_Loss, generator_image = generator_training(inputs,targets, generator_opt, L1_lambda)

    if (epoch % 10) == 0:
        print(f"After epoch {epoch + 1}, results for first 5 data:")
        print("Vegetation")
        utils.print_data(inputs, 5)
        print("Generated Bare")
        utils.print_data(generator_image, 5)
        print("Actual Bare")
        utils.print_data(targets, 5)

inputs = dataset_test[:, 1:8]
targets = dataset_test[:, 9:]
gen = generator(inputs)
veg = inputs.detach().cpu()
gen = gen.detach().cpu()
bare = targets.detach().cpu()
print("Training Done. Showing Test Results (for first 5 data).")
print("Vegetation")
utils.print_data(veg, 5)
print("Generated Bare")
utils.print_data(gen, 5)
print("Actual Bare")
utils.print_data(bare, 5)

dataset_test = dataset_test.detach().cpu()
dataset_test[:,9:] = gen

columns = ['OC', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'NDVI', 'B1_bare', 'B2_bare', 'B3_bare', 'B4_bare', 'B5_bare', 'B6_bare', 'B7_bare']
df = pd.DataFrame(dataset_test, columns=columns)
df.to_csv('data/test_generated.csv', index=False)

utils.print_metrics(gen.numpy(), bare.numpy())
