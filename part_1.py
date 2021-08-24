import torch
import matplotlib.pylab as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dsets

torch.manual_seed(0)

def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + str(data_sample[1]))

dataset = dsets.MNIST(root= './data', download=True, transform=transforms.ToTensor())

show_data(dataset[0])

