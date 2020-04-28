import os
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision import datasets, transforms


class DataLoader:
    def __init__(self):
        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5,), (0.5,)),
                                              ])
        self._path = f'{os.getcwd()}/data/'

    def get_iterable_data(self, batch_size=64, train=True):
        dataset = datasets.MNIST(self._path + ('/train' if train else '/test'),
                                 download=True,
                                 train=train,
                                 transform=self._transform)
        loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader

