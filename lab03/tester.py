import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from model import DigitRecognitionModel


class ModelTester:
    def __init__(self, model: DigitRecognitionModel):
        self._model = model
        dataloader = DataLoader()
        self._data_iterable = dataloader.get_iterable_data(train=False)

    def test_accuracy(self):
        self._model.eval()
        print("\nTesting model...")
        correct_count, all_count = 0, 0
        for images, labels in self._data_iterable:
            for i in range(len(labels)):
                pred_label = self._model.predict(images[i])
                true_label = labels.numpy()[i]
                if true_label == pred_label:
                    correct_count += 1
                all_count += 1

        print("Number of images tested: ", all_count)
        print("Model accuracy: ", (correct_count / all_count))

    def test_visually(self, images_count=5):
        images, labels = next(iter(self._data_iterable))
        for i in range(images_count):
            img = images[i]
            ps = self._model.get_probabilities_list(img)
            self.view_classify(img, ps)

    @staticmethod
    def view_classify(img, ps):
        """
        Function for viewing an image and it's predicted classes.
        """
        ps = ps.data.numpy().squeeze()
        img = img.view(1, 28, 28)

        fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
        ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
        ax1.axis('off')
        ax2.barh(np.arange(10), ps)
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(10))
        ax2.set_yticklabels(np.arange(10))
        ax2.set_title('Class Probability')
        ax2.set_xlim(0, 1.1)
        plt.tight_layout()
        plt.show()
