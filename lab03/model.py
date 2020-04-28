import os

import torch
from time import time
from torch import nn, optim

DEF_MODEL_NAME = 'mnist_model.pt'


class DigitRecognitionModel:
    def __init__(self):
        # 28px x 28px = 784
        input_size = 784
        # two hidden layers
        hidden_sizes = [128, 64]
        # output layer
        output_size = 10

        self._model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                    nn.ReLU(),
                                    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                    nn.ReLU(),
                                    nn.Linear(hidden_sizes[1], output_size),
                                    nn.LogSoftmax(dim=1))
        self._optimizer = optim.SGD(self._model.parameters(), lr=0.003, momentum=0.9)

    def train(self, data_iterable, epochs=15):
        print('\nStarted training...')
        criterion = nn.NLLLoss()

        time0 = time()
        for e in range(epochs):
            running_loss = 0
            for images, labels in data_iterable:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)

                # Training pass
                self._optimizer.zero_grad()

                output = self._model(images)
                loss = criterion(output, labels)

                # This is where the model learns by backpropagating
                loss.backward()

                # And optimizes its weights here
                self._optimizer.step()

                running_loss += loss.item()

            print("Epoch {} - Training loss: {}".format(e, running_loss / len(data_iterable)))
        print("Training time:{:.2f} minutes".format((time() - time0) / 60))

    def eval(self):
        self._model.eval()
        print("Model switched to evaluation mode (train=False)")

    def save(self, model_name=DEF_MODEL_NAME):
        path = self._get_model_path(model_name)
        state = {
            'state_dict': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }
        torch.save(state, path)
        print(f"Model saved to file '{path}'")

    def load(self, model_name=DEF_MODEL_NAME):
        path = self._get_model_path(model_name)
        state = torch.load(path)
        self._model.load_state_dict(state['state_dict'])
        self._optimizer.load_state_dict(state['optimizer'])
        print(f"Model successfully loaded from file '{path}'")

    def get_probabilities_list(self, img):
        img = img.view(1, 784)
        # Turn off gradients to speed up this part
        with torch.no_grad():
            logps = self._model(img)
        # Output of the network are log-probabilities, need to take exponential for probabilities
        return torch.exp(logps)

    def predict(self, img):
        ps = self.get_probabilities_list(img)
        probab = list(ps.numpy()[0])
        return probab.index(max(probab))

    @staticmethod
    def _get_model_path(model_name):
        return f'{os.getcwd()}/models/{model_name}'
