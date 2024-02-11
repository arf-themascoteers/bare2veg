import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def print_images(image_tensor, num_images):
    plt.show()


def normalize(data):
    for i in range(data.shape[1]):
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(data[:, i].reshape(-1, 1))
        data[:, i] = np.squeeze(x_scaled)
    return data