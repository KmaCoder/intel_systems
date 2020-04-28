from data_loader import DataLoader
from model import DigitRecognitionModel
from tester import ModelTester


def train_and_save_model():
    dataloader = DataLoader()
    data_iterable = dataloader.get_iterable_data(train=True)

    model = DigitRecognitionModel()
    model.train(data_iterable, epochs=15)
    model.save()
    return model


def load_model():
    model = DigitRecognitionModel()
    model.load()
    return model


def test_model(model: DigitRecognitionModel):
    tester = ModelTester(model)
    tester.test_accuracy()
    tester.test_visually(5)


if __name__ == "__main__":
    model = load_model()  # or train_and_save_model()
    test_model(model)
